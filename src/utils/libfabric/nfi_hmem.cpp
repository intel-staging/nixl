/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Intel Corporation. All rights reserved.
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

/*
 * Vendor table for nfi_hmem.h. Each runtime is reached through dlopen, so only its SDK headers are
 * a build-time dependency, and its section compiles out without them.
 */

#include "nfi_hmem.h"

#include <dlfcn.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#ifdef HAVE_ZE
#include <level_zero/ze_api.h>
#endif

/****************************************
 * Diagnostics
 *****************************************/

/* Atomic: a consumer may install or clear the sink while another thread logs. */
static std::atomic<nfi_hmem_log_fn> nfi_log_fn{nullptr};

void
nfi_hmem_set_log_fn(nfi_hmem_log_fn fn) {
    nfi_log_fn.store(fn, std::memory_order_release);
}

__attribute__((format(printf, 2, 3))) static void
nfi_log(enum nfi_hmem_log_level level, const char *fmt, ...) {
    /* Read once, so a concurrent clear cannot take the sink away mid-call. */
    const nfi_hmem_log_fn fn = nfi_log_fn.load(std::memory_order_acquire);
    if (fn == nullptr) {
        return;
    }

    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    fn(level, buf);
}

/****************************************
 * Vendor table shape
 *****************************************/

struct nfi_vendor {
    enum fi_hmem_iface iface;

    /* Runs once per process; false makes the vendor unavailable. nullptr without the SDK. */
    bool (*init)(void);

    /* Returns false for a pointer the runtime does not own. nullptr without the SDK. */
    bool (*query)(const void *addr, struct nfi_hmem_info *info);

    /* Enumerates accelerator PCI addresses; nullptr for a vendor that cannot. */
    size_t (*device_bdfs)(struct nfi_hmem_bus_id *out, size_t max);
};

/****************************************
 * Level Zero
 *****************************************/

#ifdef HAVE_ZE

/* Entry points resolved from libze_loader with dlsym. */
struct nfi_ze_ops {
    ze_result_t (*init)(ze_init_flags_t);
    ze_result_t (*driver_get)(uint32_t *, ze_driver_handle_t *);
    ze_result_t (*device_get)(ze_driver_handle_t, uint32_t *, ze_device_handle_t *);
    ze_result_t (*device_get_sub)(ze_device_handle_t, uint32_t *, ze_device_handle_t *);
    ze_result_t (*context_create)(ze_driver_handle_t,
                                  const ze_context_desc_t *,
                                  ze_context_handle_t *);
    ze_result_t (*device_pci_props)(ze_device_handle_t, ze_pci_ext_properties_t *);
    ze_result_t (*device_props)(ze_device_handle_t, ze_device_properties_t *);
    ze_result_t (*mem_alloc_props)(ze_context_handle_t,
                                   const void *,
                                   ze_memory_allocation_properties_t *,
                                   ze_device_handle_t *);
    /* Optional: without it the pooling check in nfi_ze_check_pooled() is skipped. */
    ze_result_t (*mem_address_range)(ze_context_handle_t, const void *, void **, size_t *);
};

/* One Level Zero device. Kept in zeDeviceGet() order, so the index is libfabric's ordinal. */
struct nfi_ze_device {
    ze_device_handle_t handle = nullptr;
    struct nfi_hmem_bus_id bus = {};
    /* Whether bus was readable; an all-zero address is valid. */
    bool has_bus = false;
    /* Not ZE_DEVICE_PROPERTY_FLAG_INTEGRATED. Only discrete GPUs are reported as accelerators. */
    bool discrete = false;
};

static struct nfi_ze_ops nfi_ze;
static ze_context_handle_t nfi_ze_context;
static std::vector<nfi_ze_device> nfi_ze_devices;

static bool
nfi_ze_load(void *handle) {
    struct {
        const char *name;
        void **slot;
    } syms[] = {
        {"zeInit", (void **)&nfi_ze.init},
        {"zeDriverGet", (void **)&nfi_ze.driver_get},
        {"zeDeviceGet", (void **)&nfi_ze.device_get},
        {"zeDeviceGetSubDevices", (void **)&nfi_ze.device_get_sub},
        {"zeContextCreate", (void **)&nfi_ze.context_create},
        {"zeDevicePciGetPropertiesExt", (void **)&nfi_ze.device_pci_props},
        {"zeDeviceGetProperties", (void **)&nfi_ze.device_props},
        {"zeMemGetAllocProperties", (void **)&nfi_ze.mem_alloc_props},
    };

    for (const auto &sym : syms) {
        *sym.slot = dlsym(handle, sym.name);
        if (*sym.slot == nullptr) {
            nfi_log(NFI_HMEM_LOG_WARN, "could not resolve Level Zero symbol %s", sym.name);
            return false;
        }
    }
    nfi_ze.mem_address_range =
        reinterpret_cast<decltype(nfi_ze.mem_address_range)>(dlsym(handle, "zeMemGetAddressRange"));
    return true;
}

static bool
nfi_ze_init(void) {
    uint32_t driver_count = 0;
    uint32_t device_count = 0;

    void *handle = dlopen("libze_loader.so.1", RTLD_NOW);
    if (handle == nullptr) {
        nfi_log(NFI_HMEM_LOG_DEBUG,
                "libze_loader.so.1 not available, Level Zero detection disabled");
        return false;
    }

    if (!nfi_ze_load(handle)) {
        return false;
    }

    /*
     * GPU_ONLY, as libfabric's hmem_ze.c does, so driver 0 here is the same driver libfabric uses
     * (flags=0 would also enumerate e.g. the NPU driver).
     */
    if (nfi_ze.init(ZE_INIT_FLAG_GPU_ONLY) != ZE_RESULT_SUCCESS) {
        nfi_log(NFI_HMEM_LOG_DEBUG, "zeInit failed, Level Zero detection disabled");
        return false;
    }

    if (nfi_ze.driver_get(&driver_count, nullptr) != ZE_RESULT_SUCCESS || driver_count == 0) {
        nfi_log(NFI_HMEM_LOG_DEBUG, "no Level Zero drivers found");
        return false;
    }
    std::vector<ze_driver_handle_t> drivers(driver_count);
    if (nfi_ze.driver_get(&driver_count, drivers.data()) != ZE_RESULT_SUCCESS) {
        nfi_log(NFI_HMEM_LOG_WARN, "zeDriverGet failed to retrieve driver handles");
        return false;
    }

    /* Driver 0 only: libfabric's Level Zero support is single-driver. */
    if (driver_count > 1) {
        nfi_log(NFI_HMEM_LOG_DEBUG,
                "found %u Level Zero drivers, using driver 0 only (libfabric is single-driver)",
                static_cast<unsigned>(driver_count));
    }
    ze_driver_handle_t driver = drivers[0];

    if (nfi_ze.device_get(driver, &device_count, nullptr) != ZE_RESULT_SUCCESS ||
        device_count == 0) {
        nfi_log(NFI_HMEM_LOG_DEBUG, "no Level Zero devices on driver 0");
        return false;
    }
    std::vector<ze_device_handle_t> handles(device_count);
    if (nfi_ze.device_get(driver, &device_count, handles.data()) != ZE_RESULT_SUCCESS) {
        nfi_log(NFI_HMEM_LOG_WARN, "zeDeviceGet failed to retrieve device handles");
        return false;
    }
    ze_context_desc_t ctx_desc = {};
    ctx_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    if (nfi_ze.context_create(driver, &ctx_desc, &nfi_ze_context) != ZE_RESULT_SUCCESS) {
        nfi_log(NFI_HMEM_LOG_WARN, "zeContextCreate failed, Level Zero detection disabled");
        nfi_ze_context = nullptr;
        return false;
    }

    nfi_ze_devices.reserve(handles.size());
    for (size_t i = 0; i < handles.size(); i++) {
        nfi_ze_device dev;
        dev.handle = handles[i];

        ze_pci_ext_properties_t pci = {};
        pci.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
        if (nfi_ze.device_pci_props(dev.handle, &pci) == ZE_RESULT_SUCCESS) {
            dev.bus.domain = static_cast<uint16_t>(pci.address.domain);
            dev.bus.bus = static_cast<uint8_t>(pci.address.bus);
            dev.bus.slot = static_cast<uint8_t>(pci.address.device);
            dev.bus.func = static_cast<uint8_t>(pci.address.function);
            dev.has_bus = true;
        } else {
            nfi_log(NFI_HMEM_LOG_DEBUG,
                    "zeDevicePciGetPropertiesExt failed for Level Zero device %u",
                    static_cast<unsigned>(i));
        }

        /*
         * Integrated vs discrete comes from the driver: both report PCI class 0x0300, and
         * compute-only parts report no display class.
         */
        ze_device_properties_t props = {};
        props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        if (nfi_ze.device_props(dev.handle, &props) == ZE_RESULT_SUCCESS) {
            dev.discrete = (props.type == ZE_DEVICE_TYPE_GPU) &&
                ((props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) == 0);
        } else {
            nfi_log(NFI_HMEM_LOG_DEBUG,
                    "zeDeviceGetProperties failed for Level Zero device %u, not treating it as "
                    "discrete",
                    static_cast<unsigned>(i));
        }

        nfi_ze_devices.push_back(dev);
    }

    nfi_log(NFI_HMEM_LOG_INFO,
            "Level Zero detection enabled with %u device(s) on driver 0",
            static_cast<unsigned>(nfi_ze_devices.size()));
    return true;
}

/*
 * Index of the root device owning @p handle, or -1. zeMemGetAllocProperties() can return a
 * sub-device (tile) handle, while zeDeviceGet() lists root devices only.
 */
static int
nfi_ze_device_index(ze_device_handle_t handle) {
    for (size_t i = 0; i < nfi_ze_devices.size(); i++) {
        if (nfi_ze_devices[i].handle == handle) {
            return static_cast<int>(i);
        }
    }

    for (size_t i = 0; i < nfi_ze_devices.size(); i++) {
        uint32_t sub_count = 0;
        if (nfi_ze.device_get_sub(nfi_ze_devices[i].handle, &sub_count, nullptr) !=
                ZE_RESULT_SUCCESS ||
            sub_count == 0) {
            continue;
        }

        std::vector<ze_device_handle_t> subs(sub_count);
        if (nfi_ze.device_get_sub(nfi_ze_devices[i].handle, &sub_count, subs.data()) !=
            ZE_RESULT_SUCCESS) {
            continue;
        }

        for (uint32_t sub = 0; sub < sub_count; sub++) {
            if (subs[sub] == handle) {
                nfi_log(NFI_HMEM_LOG_DEBUG,
                        "resolved Level Zero sub-device %u to root device %u",
                        static_cast<unsigned>(sub),
                        static_cast<unsigned>(i));
                return static_cast<int>(i);
            }
        }
    }

    return -1;
}

/*
 * Detects an allocation that is a slice of a pooled buffer object. Level Zero pools small
 * allocations made without ze_external_memory_export_desc_t, and libfabric would register them at
 * the wrong dma-buf offset. Exports the dma-buf as libfabric does and compares its size with the
 * allocation's, rounded up to the 64 KiB minimum buffer object size.
 */
static void
nfi_ze_check_pooled(const void *addr, struct nfi_hmem_info *info) {
    void *base = nullptr;
    size_t size = 0;

    if (nfi_ze.mem_address_range == nullptr ||
        nfi_ze.mem_address_range(nfi_ze_context, addr, &base, &size) != ZE_RESULT_SUCCESS ||
        size == 0) {
        return;
    }

    ze_external_memory_export_fd_t export_fd = {};
    export_fd.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD;
    export_fd.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;
    export_fd.fd = -1;
    ze_memory_allocation_properties_t props = {};
    props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
    props.pNext = &export_fd;
    ze_device_handle_t device = nullptr;
    if (nfi_ze.mem_alloc_props(nfi_ze_context, addr, &props, &device) != ZE_RESULT_SUCCESS ||
        export_fd.fd < 0) {
        /* Not exportable: nothing to compare. */
        return;
    }
    /*
     * Not closed: the driver returns the same fd to every export of a buffer object, libfabric's
     * included, and closing it breaks the registration that follows. The offset is restored.
     */
    const off_t dmabuf_size = lseek(export_fd.fd, 0, SEEK_END);
    lseek(export_fd.fd, 0, SEEK_SET);
    if (dmabuf_size <= 0) {
        return;
    }

    if (nfi_hmem_ze_is_pooled_slice(static_cast<uint64_t>(size),
                                    static_cast<uint64_t>(dmabuf_size))) {
        info->pooled_dmabuf_size = static_cast<uint64_t>(dmabuf_size);
        info->alloc_size = static_cast<uint64_t>(size);
        nfi_log(NFI_HMEM_LOG_DEBUG,
                "Level Zero allocation of %zu bytes at %p is part of a %lld-byte pooled buffer",
                size,
                base,
                static_cast<long long>(dmabuf_size));
    }
}

static bool
nfi_ze_query(const void *addr, struct nfi_hmem_info *info) {
    ze_memory_allocation_properties_t props = {};
    ze_device_handle_t device = nullptr;
    int index = 0;

    props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;

    if (nfi_ze.mem_alloc_props(nfi_ze_context, addr, &props, &device) != ZE_RESULT_SUCCESS) {
        return false;
    }

    /* As in libfabric's ze_hmem_is_addr_valid(): DEVICE, SHARED and HOST memory are all ZE. */
    if (props.type == ZE_MEMORY_TYPE_UNKNOWN) {
        return false;
    }

    if (device == nullptr) {
        /* ZE_MEMORY_TYPE_HOST belongs to no device, so there is no ordinal or address. */
        nfi_log(NFI_HMEM_LOG_DEBUG,
                "Level Zero allocation %p has no owning device (type %d), reporting ordinal 0",
                addr,
                static_cast<int>(props.type));
    } else {
        index = nfi_ze_device_index(device);
        if (index < 0) {
            /* Still claimed: a wrong ordinal is recoverable, a wrong iface is not. */
            nfi_log(NFI_HMEM_LOG_WARN,
                    "Level Zero device for %p is not in the driver 0 device list, using ordinal 0",
                    addr);
            index = 0;
        } else if (nfi_ze_devices[index].has_bus) {
            info->bus_kind = NFI_HMEM_BUS_ACCELERATOR;
            info->bus = nfi_ze_devices[index].bus;
        }
    }

    info->iface = FI_HMEM_ZE;
    info->device = static_cast<uint64_t>(fi_hmem_ze_device(0, index));
    nfi_ze_check_pooled(addr, info);
    return true;
}

static size_t
nfi_ze_device_bdfs(struct nfi_hmem_bus_id *out, size_t max) {
    size_t found = 0;

    for (const nfi_ze_device &dev : nfi_ze_devices) {
        if (!dev.discrete || !dev.has_bus) {
            continue;
        }
        if (out != nullptr && found < max) {
            out[found] = dev.bus;
        }
        found++;
    }

    return found;
}

#endif /* HAVE_ZE */

/****************************************
 * Table
 *****************************************/

/*
 * Probe order: descending fi_hmem_iface, as libfabric's ofi_get_hmem_iface() iterates, so this file
 * and providers that call it (cxi, rxm, opx) attribute a pointer to the same runtime.
 */
static const struct nfi_vendor nfi_vendors[] = {
#ifdef HAVE_ZE
    {
        FI_HMEM_ZE,
        nfi_ze_init,
        nfi_ze_query,
        nfi_ze_device_bdfs,
    },
#else
    {FI_HMEM_ZE, nullptr, nullptr, nullptr},
#endif
};

#define NFI_NUM_VENDORS (sizeof(nfi_vendors) / sizeof(nfi_vendors[0]))

/*
 * Per-vendor init state, parallel to nfi_vendors[]. The mutex serialises init only; once settled,
 * the state is read with a single acquire load.
 */
enum nfi_vendor_state : unsigned char {
    NFI_VENDOR_UNTRIED = 0,
    NFI_VENDOR_OK,
    NFI_VENDOR_UNAVAILABLE,
};

static std::mutex nfi_vendor_lock;
static std::atomic<unsigned char> nfi_vendor_state[NFI_NUM_VENDORS];

static bool
nfi_vendor_available(size_t index) {
    if (index >= NFI_NUM_VENDORS) {
        return false;
    }

    unsigned char state = nfi_vendor_state[index].load(std::memory_order_acquire);
    if (state != NFI_VENDOR_UNTRIED) {
        return state == NFI_VENDOR_OK;
    }

    const std::lock_guard<std::mutex> lock(nfi_vendor_lock);
    state = nfi_vendor_state[index].load(std::memory_order_relaxed);
    if (state == NFI_VENDOR_UNTRIED) {
        const bool ok =
            nfi_vendors[index].init && nfi_vendors[index].query && nfi_vendors[index].init();
        state = ok ? NFI_VENDOR_OK : NFI_VENDOR_UNAVAILABLE;
        nfi_vendor_state[index].store(state, std::memory_order_release);
    }
    return state == NFI_VENDOR_OK;
}

static size_t
nfi_vendor_index(enum fi_hmem_iface iface) {
    for (size_t i = 0; i < NFI_NUM_VENDORS; i++) {
        if (nfi_vendors[i].iface == iface) {
            return i;
        }
    }
    return NFI_NUM_VENDORS;
}

/****************************************
 * Public interface
 *****************************************/

bool
nfi_hmem_ze_is_pooled_slice(uint64_t alloc_size, uint64_t dmabuf_size) {
    /* Smallest buffer object Level Zero allocates; a standalone allocation is rounded up to it. */
    constexpr uint64_t min_bo_size = 64 * 1024;

    if (alloc_size == 0 || alloc_size > UINT64_MAX - (min_bo_size - 1)) {
        return false;
    }
    const uint64_t rounded = (alloc_size + min_bo_size - 1) / min_bo_size * min_bo_size;
    return dmabuf_size > rounded;
}

int
nfi_hmem_query_ptr(const void *addr, struct nfi_hmem_info *info) {
    if (!addr || !info) {
        return -EINVAL;
    }

    *info = {};
    info->iface = FI_HMEM_SYSTEM;

    for (size_t i = 0; i < NFI_NUM_VENDORS; i++) {
        if (!nfi_vendor_available(i)) {
            continue;
        }
        if (nfi_vendors[i].query(addr, info)) {
            nfi_log(NFI_HMEM_LOG_DEBUG,
                    "%s claims %p (device %llu)",
                    nfi_hmem_iface_name(nfi_vendors[i].iface),
                    addr,
                    (unsigned long long)info->device);
            return 0;
        }
        /* Discard partial output from a failed probe. */
        *info = {};
        info->iface = FI_HMEM_SYSTEM;
    }

    nfi_log(
        NFI_HMEM_LOG_DEBUG, "no accelerator runtime claimed %p, treating as system memory", addr);
    return 0;
}

void
nfi_hmem_fill_mr_attr(const struct nfi_hmem_info *info, struct fi_mr_attr *attr) {
    if (!info || !attr) {
        return;
    }

    attr->iface = info->iface;

    switch (info->iface) {
    case FI_HMEM_CUDA:
        attr->device.cuda = static_cast<int>(info->device);
        break;
    case FI_HMEM_ZE:
        /* Already encoded as fi_hmem_ze_device(0, index) by nfi_ze_query(). */
        attr->device.ze = static_cast<int>(info->device);
        break;
    case FI_HMEM_NEURON:
        attr->device.neuron = static_cast<int>(info->device);
        break;
    case FI_HMEM_ROCR:
        /* device.rocr exists from libfabric 2.5 (not 2.3 or 2.4); older versions leave the
         * union zeroed. */
#if FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION) >= FI_VERSION(2, 5)
        attr->device.rocr = static_cast<int>(info->device);
#endif
        break;
    case FI_HMEM_SYSTEM:
        /* libfabric expects device unset for host memory. */
        break;
    default:
        nfi_log(NFI_HMEM_LOG_WARN,
                "no fi_mr_attr::device arm for hmem iface %d, registering with device unset",
                static_cast<int>(info->iface));
        break;
    }
}

bool
nfi_hmem_available(void) {
    for (size_t i = 0; i < NFI_NUM_VENDORS; i++) {
        if (nfi_vendor_available(i)) {
            return true;
        }
    }
    return false;
}

bool
nfi_hmem_iface_available(enum fi_hmem_iface iface) {
    return nfi_vendor_available(nfi_vendor_index(iface));
}

enum fi_hmem_iface
nfi_hmem_primary_iface(void) {
    for (size_t i = 0; i < NFI_NUM_VENDORS; i++) {
        if (nfi_vendor_available(i)) {
            return nfi_vendors[i].iface;
        }
    }
    return FI_HMEM_SYSTEM;
}

const enum fi_hmem_iface *
nfi_hmem_ifaces(size_t *count) {
    /* Filled once; thread-safe static initialisation keeps concurrent callers from racing. */
    static const std::array<enum fi_hmem_iface, NFI_NUM_VENDORS> ifaces = []() {
        std::array<enum fi_hmem_iface, NFI_NUM_VENDORS> out{};
        for (size_t i = 0; i < NFI_NUM_VENDORS; i++) {
            out[i] = nfi_vendors[i].iface;
        }
        return out;
    }();

    if (count) {
        *count = NFI_NUM_VENDORS;
    }
    return ifaces.data();
}

/* Names from fi_tostr_r(FI_TYPE_HMEM_IFACE) without the "FI_HMEM_" prefix, cached once. */
#define NFI_IFACE_NAME_SLOTS 16
#define NFI_IFACE_NAME_MAX 32

static char nfi_iface_names[NFI_IFACE_NAME_SLOTS][NFI_IFACE_NAME_MAX];
static std::once_flag nfi_iface_names_once;

static void
nfi_init_iface_names(void) {
    static const char prefix[] = "FI_HMEM_";
    const size_t prefix_len = sizeof(prefix) - 1;
    int slot;

    for (slot = 0; slot < NFI_IFACE_NAME_SLOTS; slot++) {
        enum fi_hmem_iface iface = (enum fi_hmem_iface)slot;
        char buf[NFI_IFACE_NAME_MAX];
        const char *name = buf;

        fi_tostr_r(buf, sizeof(buf), &iface, FI_TYPE_HMEM_IFACE);
        if (!strncmp(name, prefix, prefix_len)) {
            name += prefix_len;
        }

        snprintf(nfi_iface_names[slot], NFI_IFACE_NAME_MAX, "%s", name);
    }
}

const char *
nfi_hmem_iface_name(enum fi_hmem_iface iface) {
    const int slot = static_cast<int>(iface);

    std::call_once(nfi_iface_names_once, nfi_init_iface_names);

    if (slot < 0 || slot >= NFI_IFACE_NAME_SLOTS) {
        return "Unknown";
    }
    return nfi_iface_names[slot];
}

size_t
nfi_hmem_device_bdfs(enum fi_hmem_iface iface, struct nfi_hmem_bus_id *out, size_t max) {
    size_t index = nfi_vendor_index(iface);

    if (index >= NFI_NUM_VENDORS || !nfi_vendors[index].device_bdfs) {
        return 0;
    }
    if (!nfi_vendor_available(index)) {
        return 0;
    }

    return nfi_vendors[index].device_bdfs(out, out ? max : 0);
}
