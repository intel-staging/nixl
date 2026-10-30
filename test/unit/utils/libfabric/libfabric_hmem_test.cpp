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
 * Tests for the HMEM shim (nfi_hmem.cpp) and libfabric_accel.cpp. Vendor-specific checks are
 * gated on nfi_hmem_iface_available() and log a SKIP when the runtime is absent.
 */

#include "libfabric/libfabric_accel.h"
#include "libfabric/nfi_hmem.h"
#include "common/nixl_log.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

#ifdef HAVE_ZE
#include <dirent.h>
#include <dlfcn.h>
#include <level_zero/ze_api.h>
#endif

namespace {

int failures = 0;

/** @brief Every interface the shim knows about, in probe order. */
std::vector<enum fi_hmem_iface>
knownIfaces() {
    size_t count = 0;
    const enum fi_hmem_iface *raw = nfi_hmem_ifaces(&count);
    return std::vector<enum fi_hmem_iface>(raw, raw + count);
}

void
check(bool cond, const std::string &what) {
    if (cond) {
        NIXL_INFO << "  PASS: " << what;
    } else {
        NIXL_ERROR << "  FAIL: " << what;
        ++failures;
    }
}

/****************************************
 * 0. The header's C-compatible surface
 *****************************************/

// The header's types and entry points stay C-compatible.
static_assert(std::is_standard_layout_v<nfi_hmem_bus_id>,
              "nfi_hmem_bus_id must stay a C-compatible POD");
static_assert(std::is_standard_layout_v<nfi_hmem_info>,
              "nfi_hmem_info must stay a C-compatible POD");
static_assert(std::is_trivially_copyable_v<nfi_hmem_info>,
              "nfi_hmem_info must be memcpy-able, as a C caller would expect");
static_assert(std::is_same_v<decltype(nfi_hmem_query_ptr), int(const void *, nfi_hmem_info *)>,
              "nfi_hmem_query_ptr must keep its C signature");
static_assert(std::is_same_v<decltype(nfi_hmem_iface_name), const char *(enum fi_hmem_iface)>,
              "nfi_hmem_iface_name must keep its C signature");

/****************************************
 * 1. The vendor list
 *****************************************/

void
testIfaceList() {
    NIXL_INFO << "1. Testing the shim's vendor list";

    size_t count = 0;
    const enum fi_hmem_iface *raw = nfi_hmem_ifaces(&count);
    check(raw != nullptr, "nfi_hmem_ifaces() never returns null");
    check(count > 0, "the shim knows at least one interface");

    const std::vector<enum fi_hmem_iface> ifaces(raw, raw + count);

    // FI_HMEM_SYSTEM is not a vendor.
    check(std::find(ifaces.begin(), ifaces.end(), FI_HMEM_SYSTEM) == ifaces.end(),
          "FI_HMEM_SYSTEM is not listed as a vendor");

    check(std::set<enum fi_hmem_iface>(ifaces.begin(), ifaces.end()).size() == ifaces.size(),
          "no interface is listed twice");

    // Probe order descends, matching libfabric's ofi_get_hmem_iface().
    bool descending = true;
    for (size_t i = 1; i < ifaces.size(); ++i) {
        descending = descending && (ifaces[i] < ifaces[i - 1]);
    }
    check(descending, "probe order descends, matching libfabric's own ofi_get_hmem_iface()");

    std::string names;
    for (const enum fi_hmem_iface iface : ifaces) {
        names += std::string(nfi_hmem_iface_name(iface)) + " ";
    }
    NIXL_INFO << "   probe order: " << names;
}

/****************************************
 * 2. Interface names
 *****************************************/

void
testIfaceNames() {
    NIXL_INFO << "2. Testing interface names";

    check(strcmp(nfi_hmem_iface_name(FI_HMEM_SYSTEM), "SYSTEM") == 0,
          "FI_HMEM_SYSTEM is named SYSTEM");
    // Names come from fi_tostr(), so interfaces without a vendor entry are named too.
    check(strcmp(nfi_hmem_iface_name(FI_HMEM_ROCR), "ROCR") == 0,
          "an interface with no vendor entry is still named from libfabric");
    check(strcmp(nfi_hmem_iface_name(FI_HMEM_SYNAPSEAI), "SYNAPSEAI") == 0,
          "SynapseAI is named too, though the shim has no entry for it");
    check(strcmp(nfi_hmem_iface_name(static_cast<enum fi_hmem_iface>(99)), "Unknown") == 0,
          "a value libfabric does not recognise is reported as Unknown");
    // The "FI_HMEM_" prefix is stripped.
    check(strncmp(nfi_hmem_iface_name(FI_HMEM_CUDA), "FI_HMEM_", 8) != 0,
          "the redundant FI_HMEM_ prefix is stripped");

    bool all_named = true;
    for (const enum fi_hmem_iface iface : knownIfaces()) {
        const char *name = nfi_hmem_iface_name(iface);
        if (name == nullptr || strcmp(name, "Unknown") == 0 || strcmp(name, "SYSTEM") == 0) {
            NIXL_ERROR << "  iface " << iface << " has no usable name";
            all_named = false;
        }
    }
    check(all_named, "every known interface has a name of its own");
}

/****************************************
 * 3. Availability consistency
 *****************************************/

void
testAvailability() {
    NIXL_INFO << "3. Testing availability";

    // The aggregate and per-interface availability answers agree.
    bool any = false;
    for (const enum fi_hmem_iface iface : knownIfaces()) {
        const bool avail = nfi_hmem_iface_available(iface);
        NIXL_INFO << "   " << nfi_hmem_iface_name(iface) << ": "
                  << (avail ? "available" : "unavailable");
        any = any || avail;
    }
    check(any == nfi_hmem_available(), "nfi_hmem_available() agrees with the per-iface answers");

    const enum fi_hmem_iface primary = nfi_hmem_primary_iface();
    if (any) {
        check(primary != FI_HMEM_SYSTEM, "a host with a runtime reports a non-SYSTEM primary");
        check(nfi_hmem_iface_available(primary), "the primary interface is itself available");

        // The primary is the first available interface in probe order.
        enum fi_hmem_iface expected = FI_HMEM_SYSTEM;
        for (const enum fi_hmem_iface iface : knownIfaces()) {
            if (nfi_hmem_iface_available(iface)) {
                expected = iface;
                break;
            }
        }
        check(primary == expected, "the primary is the first available interface in probe order");
    } else {
        check(primary == FI_HMEM_SYSTEM, "a host with no runtime reports SYSTEM as primary");
        NIXL_INFO << "   SKIP: no accelerator runtime on this host";
    }

    // Answers are stable across calls.
    check(nfi_hmem_available() == any, "nfi_hmem_available() is stable across calls");
    check(nfi_hmem_primary_iface() == primary, "nfi_hmem_primary_iface() is stable across calls");
}

/****************************************
 * 4. Pointer attribution
 *****************************************/

void
testQueryPtr() {
    NIXL_INFO << "4. Testing pointer attribution";

    struct nfi_hmem_info info;

    // Null arguments are rejected.
    check(nfi_hmem_query_ptr(nullptr, &info) == -EINVAL, "a null pointer is rejected");
    check(nfi_hmem_query_ptr(&info, nullptr) == -EINVAL, "a null output block is rejected");

    // Host memory is reported as FI_HMEM_SYSTEM, not an error.
    std::vector<char> host(4096, 0);
    memset(&info, 0xAB, sizeof(info));
    check(nfi_hmem_query_ptr(host.data(), &info) == 0, "host memory queries successfully");
    check(info.iface == FI_HMEM_SYSTEM, "host memory is reported as FI_HMEM_SYSTEM");
    check(info.bus_kind == NFI_HMEM_BUS_NONE, "host memory reports no PCI address");
    check(info.device == 0, "host memory reports device 0, not the 0xAB fill");

    // A stack address is host memory.
    int on_stack = 0;
    check(nfi_hmem_query_ptr(&on_stack, &info) == 0 && info.iface == FI_HMEM_SYSTEM,
          "a stack address is host memory");
}

/****************************************
 * 5. fi_mr_attr population
 *****************************************/

void
testFillMrAttr() {
    NIXL_INFO << "5. Testing fi_mr_attr population";

    struct nfi_hmem_info info;
    struct fi_mr_attr attr;

    // Fields other than iface and device are left untouched.
    memset(&attr, 0, sizeof(attr));
    attr.access = FI_REMOTE_WRITE | FI_REMOTE_READ;
    attr.requested_key = 0x1234;
    attr.offset = 0;

    memset(&info, 0, sizeof(info));
    info.iface = FI_HMEM_SYSTEM;
    nfi_hmem_fill_mr_attr(&info, &attr);
    check(attr.iface == FI_HMEM_SYSTEM, "FI_HMEM_SYSTEM is written through");
    check(attr.access == (FI_REMOTE_WRITE | FI_REMOTE_READ) && attr.requested_key == 0x1234,
          "access and requested_key are left alone");

    // Each ordinal lands unmodified in its own union arm.
    info.iface = FI_HMEM_CUDA;
    info.device = 3;
    nfi_hmem_fill_mr_attr(&info, &attr);
    check(attr.iface == FI_HMEM_CUDA && attr.device.cuda == 3,
          "a CUDA ordinal lands in device.cuda unchanged");

    info.iface = FI_HMEM_ZE;
    info.device = static_cast<uint64_t>(fi_hmem_ze_device(0, 2));
    nfi_hmem_fill_mr_attr(&info, &attr);
    check(attr.iface == FI_HMEM_ZE && attr.device.ze == fi_hmem_ze_device(0, 2),
          "a Level Zero ordinal lands in device.ze already encoded, not re-encoded");

    info.iface = FI_HMEM_NEURON;
    info.device = static_cast<uint64_t>(-1);
    nfi_hmem_fill_mr_attr(&info, &attr);
    check(attr.iface == FI_HMEM_NEURON && attr.device.neuron == -1,
          "the Neuron sentinel survives the round trip to device.neuron");

    // Null arguments are tolerated.
    nfi_hmem_fill_mr_attr(nullptr, &attr);
    nfi_hmem_fill_mr_attr(&info, nullptr);
    check(true, "null arguments are ignored rather than dereferenced");
}

/****************************************
 * 6. Device inventory
 *****************************************/

void
testDeviceBdfs() {
    NIXL_INFO << "6. Testing device inventory";

    for (const enum fi_hmem_iface iface : knownIfaces()) {
        const size_t count = nfi_hmem_device_bdfs(iface, nullptr, 0);
        NIXL_INFO << "   " << nfi_hmem_iface_name(iface) << ": " << count << " device(s)";

        if (count == 0) {
            continue;
        }

        // The count-only and filling forms agree.
        std::vector<struct nfi_hmem_bus_id> ids(count);
        const size_t written = nfi_hmem_device_bdfs(iface, ids.data(), ids.size());
        check(written == count, "the count-only and filling forms agree");

        // A short buffer is not overrun.
        std::vector<struct nfi_hmem_bus_id> one(1);
        check(nfi_hmem_device_bdfs(iface, one.data(), 1) == count,
              "a short buffer still returns the true total");

        // Addresses round-trip through hwloc's normalisation.
        bool all_ok = true;
        for (const auto &id : ids) {
            const std::string rendered = accelBusIdString(id, NFI_HMEM_BUS_ACCELERATOR);
            NIXL_INFO << "     " << rendered;
            unsigned d, b, s, f;
            if (sscanf(rendered.c_str(), "%x:%x:%x.%x", &d, &b, &s, &f) != 4 || d != id.domain ||
                b != id.bus || s != id.slot || f != id.func) {
                all_ok = false;
            }
        }
        check(all_ok, "every reported address round-trips through hwloc's normalisation");
    }

    check(nfi_hmem_device_bdfs(FI_HMEM_SYNAPSEAI, nullptr, 0) == 0,
          "an interface with no table entry reports no devices");
}

/****************************************
 * 7. PCI address rendering
 *****************************************/

void
testBusIdString() {
    NIXL_INFO << "7. Testing PCI address rendering";

    struct nfi_hmem_bus_id bus;
    bus.domain = 0;
    bus.bus = 0xad;
    bus.slot = 0;
    bus.func = 0;

    // hwloc format: domain unpadded, the rest padded.
    check(accelBusIdString(bus, NFI_HMEM_BUS_ACCELERATOR) == "0:ad:00.0",
          "the domain is unpadded and the rest padded, as hwloc writes it");

    bus.domain = 0x10;
    bus.bus = 0x3a;
    bus.slot = 0x1f;
    bus.func = 3;
    check(accelBusIdString(bus, NFI_HMEM_BUS_NIC) == "10:3a:1f.3",
          "a non-zero domain and a non-zero function render correctly");

    check(accelBusIdString(bus, NFI_HMEM_BUS_NONE).empty(),
          "NFI_HMEM_BUS_NONE renders empty, so callers need not inspect the kind");

    struct nfi_hmem_info info;
    memset(&info, 0, sizeof(info));
    info.bus_kind = NFI_HMEM_BUS_NONE;
    check(accelBusIdString(info).empty(), "an info block with no address renders empty");
}

/****************************************
 * 8. Pooled-slice decision
 *****************************************/

void
testPooledSliceDecision() {
    NIXL_INFO << "8. Testing the pooled-slice decision";

    constexpr uint64_t KiB = 1024;
    constexpr uint64_t MiB = 1024 * KiB;

    // A standalone allocation's dma-buf is its size rounded up to 64 KiB.
    check(!nfi_hmem_ze_is_pooled_slice(4 * KiB, 64 * KiB),
          "a 4 KiB allocation in its own 64 KiB buffer object is not pooled");
    check(!nfi_hmem_ze_is_pooled_slice(64 * KiB, 64 * KiB),
          "an allocation exactly filling 64 KiB is not pooled");
    check(!nfi_hmem_ze_is_pooled_slice(64 * KiB + 1, 128 * KiB),
          "64 KiB + 1 rounds up to 128 KiB, and is not pooled there");
    check(!nfi_hmem_ze_is_pooled_slice(2 * MiB + 4 * KiB, 2 * MiB + 64 * KiB),
          "an odd large size rounds up to the next 64 KiB, not to 2 MiB");

    // Anything larger than the rounded size is shared with other allocations.
    check(nfi_hmem_ze_is_pooled_slice(4 * KiB, 64 * KiB + 1),
          "one byte past the rounded size is pooled");
    check(nfi_hmem_ze_is_pooled_slice(64 * KiB, 128 * KiB),
          "64 KiB inside a 128 KiB buffer object is pooled");
    check(nfi_hmem_ze_is_pooled_slice(4 * KiB, 2 * MiB), "a 4 KiB slice of a 2 MiB pool is pooled");
    check(nfi_hmem_ze_is_pooled_slice(1 * MiB, 16 * MiB),
          "a 1 MiB slice of a 16 MiB pool is pooled");

    // Degenerate inputs never claim pooling.
    check(!nfi_hmem_ze_is_pooled_slice(0, 2 * MiB), "a zero-size allocation is not pooled");
    check(!nfi_hmem_ze_is_pooled_slice(4 * KiB, 0), "a zero-size dma-buf is not pooled");
    check(!nfi_hmem_ze_is_pooled_slice(UINT64_MAX, UINT64_MAX),
          "a size whose rounding would overflow is not pooled");
}

/****************************************
 * 9. Level Zero device memory
 *****************************************/

#ifdef HAVE_ZE

/** The Level Zero entry points the test allocates with, from the loader nfi_hmem also opens. */
struct ZeApi {
    decltype(&zeInit) init = nullptr;
    decltype(&zeDriverGet) driver_get = nullptr;
    decltype(&zeDeviceGet) device_get = nullptr;
    decltype(&zeContextCreate) context_create = nullptr;
    decltype(&zeContextDestroy) context_destroy = nullptr;
    decltype(&zeMemAllocDevice) mem_alloc_device = nullptr;
    decltype(&zeMemFree) mem_free = nullptr;

    bool
    load() {
        void *handle = dlopen("libze_loader.so.1", RTLD_NOW);
        if (handle == nullptr) {
            return false;
        }
        init = reinterpret_cast<decltype(init)>(dlsym(handle, "zeInit"));
        driver_get = reinterpret_cast<decltype(driver_get)>(dlsym(handle, "zeDriverGet"));
        device_get = reinterpret_cast<decltype(device_get)>(dlsym(handle, "zeDeviceGet"));
        context_create =
            reinterpret_cast<decltype(context_create)>(dlsym(handle, "zeContextCreate"));
        context_destroy =
            reinterpret_cast<decltype(context_destroy)>(dlsym(handle, "zeContextDestroy"));
        mem_alloc_device =
            reinterpret_cast<decltype(mem_alloc_device)>(dlsym(handle, "zeMemAllocDevice"));
        mem_free = reinterpret_cast<decltype(mem_free)>(dlsym(handle, "zeMemFree"));
        return init && driver_get && device_get && context_create && context_destroy &&
            mem_alloc_device && mem_free;
    }
};

int
countOpenFds() {
    int count = 0;
    DIR *dir = opendir("/proc/self/fd");
    if (dir == nullptr) {
        return -1;
    }
    while (readdir(dir) != nullptr) {
        ++count;
    }
    closedir(dir);
    return count;
}

void *
zeAlloc(const ZeApi &ze,
        ze_context_handle_t ctx,
        ze_device_handle_t device,
        size_t size,
        bool exportable) {
    ze_external_memory_export_desc_t export_desc = {};
    export_desc.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC;
    export_desc.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;
    ze_device_mem_alloc_desc_t alloc_desc = {};
    alloc_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    alloc_desc.pNext = exportable ? &export_desc : nullptr;

    void *ptr = nullptr;
    if (ze.mem_alloc_device(ctx, &alloc_desc, size, 64, device, &ptr) != ZE_RESULT_SUCCESS) {
        return nullptr;
    }
    return ptr;
}

void
testLevelZeroMemory() {
    NIXL_INFO << "9. Testing Level Zero device memory";

    if (!nfi_hmem_iface_available(FI_HMEM_ZE)) {
        NIXL_INFO << "   SKIP: Level Zero is not usable on this host";
        return;
    }
    ZeApi ze;
    uint32_t driver_count = 1;
    ze_driver_handle_t driver = nullptr;
    uint32_t device_count = 0;
    if (!ze.load() || ze.init(ZE_INIT_FLAG_GPU_ONLY) != ZE_RESULT_SUCCESS ||
        ze.driver_get(&driver_count, &driver) != ZE_RESULT_SUCCESS || driver_count == 0 ||
        ze.device_get(driver, &device_count, nullptr) != ZE_RESULT_SUCCESS || device_count == 0) {
        check(false, "Level Zero is usable to the shim, so the test can reach it too");
        return;
    }
    std::vector<ze_device_handle_t> devices(device_count);
    ze.device_get(driver, &device_count, devices.data());

    ze_context_desc_t ctx_desc = {};
    ctx_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    ze_context_handle_t ctx = nullptr;
    if (ze.context_create(driver, &ctx_desc, &ctx) != ZE_RESULT_SUCCESS) {
        check(false, "a Level Zero context can be created");
        return;
    }

    // Each device's memory carries that device's ordinal, as libfabric numbers driver 0's devices.
    for (uint32_t i = 0; i < device_count; ++i) {
        void *ptr = zeAlloc(ze, ctx, devices[i], 64 * 1024, true);
        if (ptr == nullptr) {
            NIXL_INFO << "   device " << i << ": allocation failed, skipping it";
            continue;
        }
        struct nfi_hmem_info info;
        check(nfi_hmem_query_ptr(ptr, &info) == 0 && info.iface == FI_HMEM_ZE,
              "device " + std::to_string(i) + " memory is claimed by Level Zero");
        check(info.device == static_cast<uint64_t>(fi_hmem_ze_device(0, static_cast<int>(i))),
              "device " + std::to_string(i) + " memory carries ordinal " + std::to_string(i));
        check(info.pooled_dmabuf_size == 0,
              "an exportable allocation on device " + std::to_string(i) + " is not pooled");
        ze.mem_free(ctx, ptr);
    }

    // An interior pointer belongs to the allocation, which is still not pooled.
    void *big = zeAlloc(ze, ctx, devices[0], 1024 * 1024, true);
    if (big != nullptr) {
        struct nfi_hmem_info info;
        check(nfi_hmem_query_ptr(static_cast<char *>(big) + 4096, &info) == 0 &&
                  info.iface == FI_HMEM_ZE && info.pooled_dmabuf_size == 0,
              "an interior pointer of an exportable allocation is ZE and not pooled");

        // Each query exports the dma-buf; the driver hands back the same fd every time.
        const int fds_before = countOpenFds();
        for (int q = 0; q < 20; ++q) {
            nfi_hmem_query_ptr(big, &info);
        }
        const int fds_after = countOpenFds();
        check(fds_before >= 0 && fds_after - fds_before <= 1,
              "repeated queries of one allocation open at most one fd");
        ze.mem_free(ctx, big);
    }

    // Small allocations without an export descriptor. Whether the driver pools them is decided
    // from where they land: two in the same 64 KiB share a buffer object.
    void *a = zeAlloc(ze, ctx, devices[0], 4096, false);
    void *b = zeAlloc(ze, ctx, devices[0], 4096, false);
    if (a != nullptr && b != nullptr) {
        const uintptr_t pa = reinterpret_cast<uintptr_t>(a);
        const uintptr_t pb = reinterpret_cast<uintptr_t>(b);
        const bool pooled = (pa > pb ? pa - pb : pb - pa) < 64 * 1024;
        struct nfi_hmem_info info_a;
        struct nfi_hmem_info info_b;
        nfi_hmem_query_ptr(a, &info_a);
        nfi_hmem_query_ptr(b, &info_b);
        if (pooled) {
            check(info_a.pooled_dmabuf_size > 0 && info_b.pooled_dmabuf_size > 0,
                  "allocations sharing a buffer object are both reported as pooled");
            check(info_a.alloc_size == 4096 && info_b.alloc_size == 4096,
                  "a pooled report carries the allocation's own size");
        } else {
            NIXL_INFO << "   the driver did not pool these allocations";
            check(info_a.pooled_dmabuf_size == 0 && info_b.pooled_dmabuf_size == 0,
                  "allocations in their own buffer objects are not reported as pooled");
        }
    } else {
        check(false, "two 4 KiB device allocations succeed");
    }
    if (a != nullptr) {
        ze.mem_free(ctx, a);
    }
    if (b != nullptr) {
        ze.mem_free(ctx, b);
    }

    ze.context_destroy(ctx);
}

#endif // HAVE_ZE

} // namespace

int
main() {
#ifdef HAVE_ZE
    // Ask the Intel driver to pool small device allocations, so section 9 exercises the pooled
    // path; read at Level Zero initialisation, so set before anything initialises it. Not
    // overwritten when already set.
    setenv("NEOReadDebugKeys", "1", 0);
    setenv("EnableDeviceUsmAllocationPool", "2", 0);
#endif

    NIXL_INFO << "=== Testing the libfabric HMEM shim ===";

    // Before vendor initialisation, so its log lines appear.
    nixlAccelInstallLogBridge();

    testIfaceList();
    testIfaceNames();
    testAvailability();
    testQueryPtr();
    testFillMrAttr();
    testDeviceBdfs();
    testBusIdString();
    testPooledSliceDecision();
#ifdef HAVE_ZE
    testLevelZeroMemory();
#endif

    if (failures != 0) {
        NIXL_ERROR << "=== " << failures << " check(s) FAILED ===";
        return 1;
    }
    NIXL_INFO << "=== Test completed successfully! ===";
    return 0;
}
