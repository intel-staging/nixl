/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 Amazon.com, Inc. and affiliates.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "libfabric_accel.h"
#include "common/nixl_log.h"

#include <rdma/fi_errno.h>
#include <rdma/fi_ext.h>

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <set>

#ifdef HAVE_CUDA
#include <cuda.h>
#endif

namespace {

/** The first libfabric release carrying the hmem query extension. */
constexpr uint32_t hmem_ext_version = FI_VERSION(2, 8);

/****************************************
 * libfabric hmem extension
 *****************************************/

/**
 * @brief The process-wide hmem object, or nullptr when this libfabric lacks the extension.
 *
 * Opened once and never closed: the object is stateless and shared inside libfabric, and opening
 * it runs fi_ini(), which is what initialises libfabric's vendor runtimes.
 */
struct fid_hmem *
hmemFid() {
    static struct fid_hmem *const fid = []() -> struct fid_hmem * {
        struct fid_hmem *hmem = nullptr;
        const int ret = fi_open_hmem(hmem_ext_version, 0, &hmem, nullptr);
        if (ret) {
            NIXL_WARN << "libfabric hmem query extension unavailable (" << fi_strerror(-ret)
                      << "), device memory support disabled";
            return nullptr;
        }
        return hmem;
    }();
    return fid;
}

bool
hmemIfaceEnabled(enum fi_hmem_iface iface) {
    struct fid_hmem *hmem = hmemFid();
    return hmem && fi_hmem_iface_enabled(hmem, iface);
}

/****************************************
 * Neuron
 *****************************************/

using nrtGetAttachedEfaBdfFn = int (*)(const void *va, char *efa_bdf, size_t *len);

/**
 * @brief libnrt's pointer-to-EFA lookup, or nullptr when Neuron is not usable.
 *
 * Usable means both that libnrt resolves and that libfabric initialised its own Neuron support,
 * since a pointer libfabric cannot register as FI_HMEM_NEURON is no use to claim.
 */
nrtGetAttachedEfaBdfFn
nrtGetAttachedEfaBdf() {
    static const nrtGetAttachedEfaBdfFn fn = []() -> nrtGetAttachedEfaBdfFn {
        if (!hmemIfaceEnabled(FI_HMEM_NEURON)) {
            return nullptr;
        }
        void *handle = dlopen("libnrt.so.1", RTLD_NOW);
        if (!handle) {
            NIXL_DEBUG << "libnrt.so.1 not available, Neuron detection disabled";
            return nullptr;
        }
        auto sym =
            reinterpret_cast<nrtGetAttachedEfaBdfFn>(dlsym(handle, "nrt_get_attached_efa_bdf"));
        if (!sym) {
            NIXL_WARN << "could not resolve libnrt symbol nrt_get_attached_efa_bdf";
        }
        return sym;
    }();
    return fn;
}

bool
neuronQueryPtr(const void *addr, nixlAccelInfo &info) {
    const nrtGetAttachedEfaBdfFn get_bdf = nrtGetAttachedEfaBdf();
    char buf[] = "0000:00:00.0";
    size_t buflen = sizeof(buf);
    unsigned domain, bus, slot, func;

    if (!get_bdf || get_bdf(addr, buf, &buflen) != 0) {
        return false;
    }

    info.iface = FI_HMEM_NEURON;
    // libfabric requires fi_mr_attr::device to be initialised for Neuron but EFA never reads it;
    // the -1 sentinel follows the spec and would fault loudly if a provider started to.
    info.device = static_cast<uint64_t>(-1);

    if (sscanf(buf, "%x:%x:%x.%x", &domain, &bus, &slot, &func) == 4) {
        info.bus_kind = nixlAccelBusKind::NIC;
        info.bus.domain_id = static_cast<uint16_t>(domain);
        info.bus.bus_id = static_cast<uint8_t>(bus);
        info.bus.device_id = static_cast<uint8_t>(slot);
        info.bus.function_id = static_cast<uint8_t>(func);
    }
    return true;
}

/****************************************
 * hwloc predicates
 *****************************************/

/** @brief Whether @p obj is a PCI device from @p vendor_id in the display/3D class range. */
bool
isDisplayClassPciDev(hwloc_obj_t obj, uint16_t vendor_id) {
    if (!obj || obj->type != HWLOC_OBJ_PCI_DEVICE) {
        return false;
    }
    if (obj->attr->pcidev.vendor_id != vendor_id) {
        return false;
    }
    // 0x300-0x3ff are display controllers; 0x302 is "3D controller", which is what a
    // compute-oriented NVIDIA board reports.
    const uint16_t class_id = obj->attr->pcidev.class_id;
    return (class_id >= 0x300 && class_id < 0x400);
}

/**
 * @brief Whether @p obj is an NVIDIA GPU. Pure hwloc; see accelIfaceOfHwlocObj for why.
 */
bool
isNvidiaAccel(hwloc_obj_t obj) {
    return isDisplayClassPciDev(obj, 0x10de);
}

/**
 * @brief Whether @p obj is a Neuron accelerator, by vendor ID and device ID.
 *
 * Pure hwloc, so Trainium cards are counted in a build without libnrt. The device-ID allowlist is
 * the cost of that: unlike a GPU, a Neuron card has no PCI class separating it from the other
 * Annapurna devices sharing vendor 0x1d0f, so each generation has to be added here. libnrt exposes
 * no device enumeration to replace it with -- nrt_get_attached_efa_bdf() answers only about a
 * pointer -- which is why this is the one vendor with no SDK-side alternative.
 */
bool
isNeuronAccel(hwloc_obj_t obj) {
    if (!obj || obj->type != HWLOC_OBJ_PCI_DEVICE) {
        return false;
    }
    // Amazon vendor ID is 0x1d0f
    if (obj->attr->pcidev.vendor_id != 0x1d0f) {
        return false;
    }
    static const uint16_t NEURON_DEVICE_IDS[] = {
        0x7264, // INF2
        0x7164, // TRN1
        0x7364, // TRN2
        0x7564, // TRN3_DEVICE_0
        0x7565, // TRN3_DEVICE_1
    };
    return std::find(std::begin(NEURON_DEVICE_IDS),
                     std::end(NEURON_DEVICE_IDS),
                     obj->attr->pcidev.device_id) != std::end(NEURON_DEVICE_IDS);
}

/**
 * @brief The discrete Level Zero GPUs libfabric enumerated, in hwloc's normalisation.
 *
 * Integrated GPUs are left out: memory on one registers perfectly well, but a device on the CPU
 * package has no meaningful PCIe distance to a NIC and must not enter proximity grouping. Cached
 * because the only caller walks every PCI device in the topology.
 */
const std::set<std::string> &
zeAccelBdfs() {
    static const std::set<std::string> bdfs = []() {
        std::set<std::string> out;
        struct fid_hmem *hmem = hmemFid();
        size_t count = 0;

        if (!hmem || fi_hmem_dev_count(hmem, FI_HMEM_ZE, &count)) {
            return out;
        }

        for (size_t i = 0; i < count; ++i) {
            struct fi_hmem_dev_attr dev;
            if (fi_hmem_dev_attr(hmem, FI_HMEM_ZE, i, &dev) == 0 &&
                dev.type == FI_HMEM_DEV_DISCRETE && dev.bus.bus_type == FI_BUS_PCI) {
                out.insert(accelBusIdString(dev.bus.attr.pci));
            }
        }
        return out;
    }();
    return bdfs;
}

/**
 * @brief Whether @p obj is a Level-Zero-capable discrete Intel accelerator.
 *
 * Membership in the driver's own list is the entire test -- no PCI class range, no per-generation
 * device-ID allowlist. Both alternatives are wrong in one direction or the other: a discrete Intel
 * GPU and Intel integrated graphics share class 0x0300 so the class cannot separate them, while
 * compute-only parts carry no display class at all and a class check silently drops them.
 */
bool
isIntelAccel(hwloc_obj_t obj) {
    if (!obj || obj->type != HWLOC_OBJ_PCI_DEVICE) {
        return false;
    }

    const std::set<std::string> &enumerated = zeAccelBdfs();
    if (enumerated.empty()) {
        return false;
    }

    struct fi_pci_attr bus;
    bus.domain_id = static_cast<uint16_t>(obj->attr->pcidev.domain);
    bus.bus_id = static_cast<uint8_t>(obj->attr->pcidev.bus);
    bus.device_id = static_cast<uint8_t>(obj->attr->pcidev.dev);
    bus.function_id = static_cast<uint8_t>(obj->attr->pcidev.func);

    return enumerated.count(accelBusIdString(bus)) > 0;
}

/****************************************
 * CUDA context binding
 *****************************************/

#ifdef HAVE_CUDA
/**
 * @brief Whether CUDA contexts can be bound: libfabric enabled CUDA and the driver initialises.
 *
 * cuInit is idempotent and libfabric has already initialised the driver by the time it reports
 * CUDA enabled, so this costs nothing beyond the first call.
 */
bool
cudaScopeAvailable() {
    static const bool available =
        accelIfaceAvailable(FI_HMEM_CUDA) && cuInit(0) == CUDA_SUCCESS;
    return available;
}
#endif

} // namespace

/****************************************
 * Pointer attribution
 *****************************************/

nixl_status_t
accelQueryPtr(const void *addr, nixlAccelInfo &info) {
    if (!addr) {
        return NIXL_ERR_INVALID_PARAM;
    }

    info = nixlAccelInfo{};

    // Neuron first, matching libfabric's descending probe order: libfabric itself has no Neuron
    // probe, so the remaining interfaces are left to it.
    if (neuronQueryPtr(addr, info)) {
        return NIXL_SUCCESS;
    }
    info = nixlAccelInfo{};

    struct fid_hmem *hmem = hmemFid();
    if (!hmem) {
        return NIXL_SUCCESS;
    }

    struct fi_hmem_addr_attr attr;
    const int ret = fi_hmem_query_addr(hmem, addr, &attr);
    if (ret) {
        NIXL_ERROR << "fi_hmem_query_addr failed for " << addr << ": " << fi_strerror(-ret);
        return NIXL_ERR_BACKEND;
    }

    info.iface = attr.iface;
    info.device = attr.device;

    struct fi_hmem_dev_attr dev;
    if (attr.iface != FI_HMEM_SYSTEM &&
        fi_hmem_dev_lookup(hmem, attr.iface, attr.device, &dev) == 0 &&
        dev.bus.bus_type == FI_BUS_PCI) {
        info.bus_kind = nixlAccelBusKind::ACCELERATOR;
        info.bus = dev.bus.attr.pci;
    }
    return NIXL_SUCCESS;
}

void
accelFillMrAttr(const nixlAccelInfo &info, struct fi_mr_attr &attr) {
    struct fid_hmem *hmem = hmemFid();

    // Without the extension nothing can have been attributed, so the info block is host memory.
    if (!hmem || fi_hmem_set_mr_attr(hmem, &attr, info.iface, info.device)) {
        attr.iface = FI_HMEM_SYSTEM;
    }
}

/****************************************
 * Runtime enablement
 *****************************************/

const std::vector<enum fi_hmem_iface> &
nixlAccelKnownIfaces() {
    static const std::vector<enum fi_hmem_iface> ifaces = {
        FI_HMEM_NEURON,
        FI_HMEM_ZE,
        FI_HMEM_CUDA,
    };
    return ifaces;
}

bool
accelIfaceAvailable(enum fi_hmem_iface iface) {
    if (iface == FI_HMEM_NEURON) {
        return nrtGetAttachedEfaBdf() != nullptr;
    }
    return hmemIfaceEnabled(iface);
}

bool
accelAvailable() {
    return accelPrimaryIface() != FI_HMEM_SYSTEM;
}

enum fi_hmem_iface
accelPrimaryIface() {
    for (const enum fi_hmem_iface iface : nixlAccelKnownIfaces()) {
        if (accelIfaceAvailable(iface)) {
            return iface;
        }
    }
    return FI_HMEM_SYSTEM;
}

/*
 * Names come from libfabric rather than a table of our own, with the redundant "FI_HMEM_" prefix
 * stripped. Built once because fi_tostr() writes a shared static buffer.
 */
const char *
accelIfaceName(enum fi_hmem_iface iface) {
    static const std::array<std::string, 16> names = []() {
        std::array<std::string, 16> out;
        for (size_t i = 0; i < out.size(); ++i) {
            enum fi_hmem_iface value = static_cast<enum fi_hmem_iface>(i);
            char buf[32];
            fi_tostr_r(buf, sizeof(buf), &value, FI_TYPE_HMEM_IFACE);
            const char *name = strncmp(buf, "FI_HMEM_", 8) ? buf : buf + 8;
            out[i] = name;
        }
        return out;
    }();

    const size_t slot = static_cast<size_t>(iface);
    return slot < names.size() ? names[slot].c_str() : "Unknown";
}

/****************************************
 * PCI addresses
 *****************************************/

std::string
accelBusIdString(const struct fi_pci_attr &bus) {
    char buf[32];
    snprintf(buf,
             sizeof(buf),
             "%x:%02x:%02x.%x",
             static_cast<unsigned>(bus.domain_id),
             static_cast<unsigned>(bus.bus_id),
             static_cast<unsigned>(bus.device_id),
             static_cast<unsigned>(bus.function_id));
    return std::string(buf);
}

std::string
accelBusIdString(const nixlAccelInfo &info) {
    return info.bus_kind == nixlAccelBusKind::NONE ? std::string() : accelBusIdString(info.bus);
}

/****************************************
 * hwloc matching and policy
 *****************************************/

enum fi_hmem_iface
accelIfaceOfHwlocObj(hwloc_obj_t obj) {
    if (isNvidiaAccel(obj)) {
        return FI_HMEM_CUDA;
    }
    if (isNeuronAccel(obj)) {
        return FI_HMEM_NEURON;
    }
    if (isIntelAccel(obj)) {
        return FI_HMEM_ZE;
    }
    return FI_HMEM_SYSTEM;
}

bool
accelIfaceIsGroupable(enum fi_hmem_iface iface) {
    switch (iface) {
    case FI_HMEM_CUDA:
    case FI_HMEM_ZE:
        return true;
    case FI_HMEM_NEURON:
        // A Trainium card carries its EFA device onboard, so a Neuron pointer reports the NIC's
        // address and resolves through the NIC map. Admitting it to grouping would look up an
        // accelerator that is not there.
        return false;
    default:
        return false;
    }
}

bool
accelRequiresStrictAttribution() {
    // CUDA only: on a CUDA host, device memory cuPointerGetAttributes() does not recognise is a
    // caller bug. Asked of the runtime rather than the topology so the answer does not depend on
    // which libfabric provider was selected.
    return accelIfaceAvailable(FI_HMEM_CUDA);
}

/****************************************
 * Context scope
 *****************************************/

nixlAccelScope::nixlAccelScope(enum fi_hmem_iface iface, uint64_t device) {
#ifdef HAVE_CUDA
    if (iface != FI_HMEM_CUDA || !cudaScopeAvailable()) {
        return;
    }

    // The primary context always exists and cannot go stale, so there is nothing to discover.
    const CUdevice dev = static_cast<CUdevice>(device);
    CUcontext primary = nullptr;
    if (cuDevicePrimaryCtxRetain(&primary, dev) != CUDA_SUCCESS) {
        NIXL_ERROR << "cuDevicePrimaryCtxRetain failed for CUDA device " << dev;
        ok_ = false;
        return;
    }
    if (cuCtxPushCurrent(primary) != CUDA_SUCCESS) {
        (void)cuDevicePrimaryCtxRelease(dev);
        NIXL_ERROR << "cuCtxPushCurrent failed for CUDA device " << dev;
        ok_ = false;
        return;
    }
    device_ = dev;
    pushed_ = true;
#else
    (void)iface;
    (void)device;
#endif
}

nixlAccelScope::~nixlAccelScope() {
#ifdef HAVE_CUDA
    if (!pushed_) {
        return;
    }
    CUcontext popped = nullptr;
    if (cuCtxPopCurrent(&popped) != CUDA_SUCCESS) {
        NIXL_WARN << "cuCtxPopCurrent failed";
    }
    if (cuDevicePrimaryCtxRelease(static_cast<CUdevice>(device_)) != CUDA_SUCCESS) {
        NIXL_WARN << "cuDevicePrimaryCtxRelease failed for CUDA device " << device_;
    }
#endif
}
