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

/*
 * Tests for nixl's side of the accelerator boundary (src/utils/libfabric/libfabric_accel.cpp),
 * which wraps libfabric's hmem query extension.
 *
 * Written to pass on any host: with no accelerator, with an NVIDIA GPU, with a Neuron device, and
 * with an Intel Xe GPU. Assertions that need a particular vendor are gated on
 * accelIfaceAvailable() and skipped with a visible log line, so a skip never reads as a pass.
 */

#include "libfabric/libfabric_accel.h"
#include "common/nixl_log.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#ifdef CUDA_FOUND
#include <cuda.h>
#endif

namespace {

int failures = 0;

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
 * 1. The interface list
 *****************************************/

void
testIfaceList() {
    NIXL_INFO << "1. Testing the interface list";

    const std::vector<enum fi_hmem_iface> &ifaces = nixlAccelKnownIfaces();
    check(!ifaces.empty(), "at least one interface is known");

    // FI_HMEM_SYSTEM is not a vendor; listing it would make it a candidate primary runtime.
    check(std::find(ifaces.begin(), ifaces.end(), FI_HMEM_SYSTEM) == ifaces.end(),
          "FI_HMEM_SYSTEM is not listed as a vendor");
    check(std::set<enum fi_hmem_iface>(ifaces.begin(), ifaces.end()).size() == ifaces.size(),
          "no interface is listed twice");

    // Descending, matching libfabric's own probe order, so the primary runtime is the one
    // libfabric would claim a pointer for first on a host with two runtimes.
    bool descending = true;
    for (size_t i = 1; i < ifaces.size(); ++i) {
        descending = descending && (ifaces[i] < ifaces[i - 1]);
    }
    check(descending, "the list descends, matching libfabric's probe order");
}

/****************************************
 * 2. Interface names
 *****************************************/

void
testIfaceNames() {
    NIXL_INFO << "2. Testing interface names";

    check(strcmp(accelIfaceName(FI_HMEM_SYSTEM), "SYSTEM") == 0, "FI_HMEM_SYSTEM is named SYSTEM");
    // Names come from libfabric, so interfaces nixl does not handle are still named correctly.
    check(strcmp(accelIfaceName(FI_HMEM_ROCR), "ROCR") == 0, "ROCR is named from libfabric");
    check(strcmp(accelIfaceName(FI_HMEM_SYNAPSEAI), "SYNAPSEAI") == 0,
          "SynapseAI is named from libfabric");
    check(strcmp(accelIfaceName(static_cast<enum fi_hmem_iface>(99)), "Unknown") == 0,
          "an out-of-range value is reported as Unknown");
    check(strncmp(accelIfaceName(FI_HMEM_CUDA), "FI_HMEM_", 8) != 0,
          "the redundant FI_HMEM_ prefix is stripped");

    bool all_named = true;
    for (const enum fi_hmem_iface iface : nixlAccelKnownIfaces()) {
        const char *name = accelIfaceName(iface);
        if (strcmp(name, "Unknown") == 0 || strcmp(name, "SYSTEM") == 0) {
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

    // Consistency, not a specific answer: getSupportedMems() and the registerMemory() fallback
    // must agree about whether VRAM works.
    enum fi_hmem_iface expected = FI_HMEM_SYSTEM;
    for (const enum fi_hmem_iface iface : nixlAccelKnownIfaces()) {
        const bool avail = accelIfaceAvailable(iface);
        NIXL_INFO << "   " << accelIfaceName(iface) << ": " << (avail ? "available" : "unavailable");
        if (avail && expected == FI_HMEM_SYSTEM) {
            expected = iface;
        }
    }

    check(accelAvailable() == (expected != FI_HMEM_SYSTEM),
          "accelAvailable() agrees with the per-iface answers");
    check(accelPrimaryIface() == expected,
          "the primary interface is the first available in probe order");
    if (expected == FI_HMEM_SYSTEM) {
        NIXL_INFO << "   SKIP: no accelerator runtime on this host";
    }

    check(accelPrimaryIface() == expected, "the primary interface is stable across calls");
}

/****************************************
 * 4. Pointer attribution
 *****************************************/

void
testQueryPtr() {
    NIXL_INFO << "4. Testing pointer attribution";

    nixlAccelInfo info;
    check(accelQueryPtr(nullptr, info) == NIXL_ERR_INVALID_PARAM, "a null pointer is rejected");

    // Host memory is a successful FI_HMEM_SYSTEM answer, not an error, or every DRAM registration
    // would look like a failure.
    std::vector<char> host(4096, 0);
    info.device = 0xABAB;
    info.bus_kind = nixlAccelBusKind::NIC;
    check(accelQueryPtr(host.data(), info) == NIXL_SUCCESS, "host memory queries successfully");
    check(info.iface == FI_HMEM_SYSTEM, "host memory is reported as FI_HMEM_SYSTEM");
    check(info.bus_kind == nixlAccelBusKind::NONE, "host memory reports no PCI address");
    check(info.device == 0, "a previous answer does not leak into the result");

    int on_stack = 0;
    check(accelQueryPtr(&on_stack, info) == NIXL_SUCCESS && info.iface == FI_HMEM_SYSTEM,
          "a stack address is host memory");
}

/****************************************
 * 5. fi_mr_attr population
 *****************************************/

void
testFillMrAttr() {
    NIXL_INFO << "5. Testing fi_mr_attr population";

    nixlAccelInfo info;
    struct fi_mr_attr attr;

    memset(&attr, 0, sizeof(attr));
    attr.access = FI_REMOTE_WRITE | FI_REMOTE_READ;
    attr.requested_key = 0x1234;

    accelFillMrAttr(info, attr);
    check(attr.iface == FI_HMEM_SYSTEM, "FI_HMEM_SYSTEM is written through");
    check(attr.access == (FI_REMOTE_WRITE | FI_REMOTE_READ) && attr.requested_key == 0x1234,
          "access and requested_key are left alone");

    // Each ordinal must land in its own union arm unmodified; re-encoding would double-encode
    // the Level Zero ordinal.
    info.iface = FI_HMEM_CUDA;
    info.device = 3;
    accelFillMrAttr(info, attr);
    check(attr.iface == FI_HMEM_CUDA && attr.device.cuda == 3,
          "a CUDA ordinal lands in device.cuda unchanged");

    info.iface = FI_HMEM_ZE;
    info.device = static_cast<uint64_t>(fi_hmem_ze_device(0, 2));
    accelFillMrAttr(info, attr);
    check(attr.iface == FI_HMEM_ZE && attr.device.ze == fi_hmem_ze_device(0, 2),
          "a Level Zero ordinal lands in device.ze already encoded");

    info.iface = FI_HMEM_NEURON;
    info.device = static_cast<uint64_t>(-1);
    accelFillMrAttr(info, attr);
    check(attr.iface == FI_HMEM_NEURON && attr.device.neuron == -1,
          "the Neuron sentinel survives the round trip to device.neuron");
}

/****************************************
 * 6. hwloc matching
 *****************************************/

void
testHwlocMatching() {
    NIXL_INFO << "6. Testing hwloc matching";

    check(accelIfaceOfHwlocObj(nullptr) == FI_HMEM_SYSTEM,
          "a null hwloc object is rejected, not dereferenced");

    hwloc_topology_t topo;
    if (hwloc_topology_init(&topo) != 0) {
        check(false, "hwloc topology initialises");
        return;
    }
    hwloc_topology_set_io_types_filter(topo, HWLOC_TYPE_FILTER_KEEP_ALL);
    if (hwloc_topology_load(topo) != 0) {
        hwloc_topology_destroy(topo);
        check(false, "hwloc topology loads");
        return;
    }

    // Every object attributed to a vendor must carry that vendor's PCI vendor ID; Level Zero
    // attribution comes from libfabric's device list, so this also checks that the PCI addresses
    // libfabric reports render the way hwloc keys them.
    const std::map<enum fi_hmem_iface, uint16_t> vendor_ids = {
        {FI_HMEM_CUDA, 0x10de}, {FI_HMEM_NEURON, 0x1d0f}, {FI_HMEM_ZE, 0x8086}};
    std::map<enum fi_hmem_iface, int> counts;
    bool vendors_match = true;

    for (hwloc_obj_t obj = hwloc_get_next_pcidev(topo, nullptr); obj;
         obj = hwloc_get_next_pcidev(topo, obj)) {
        const enum fi_hmem_iface iface = accelIfaceOfHwlocObj(obj);
        if (iface == FI_HMEM_SYSTEM) {
            continue;
        }
        counts[iface]++;
        if (obj->attr->pcidev.vendor_id != vendor_ids.at(iface)) {
            vendors_match = false;
        }
        struct fi_pci_attr bus;
        bus.domain_id = static_cast<uint16_t>(obj->attr->pcidev.domain);
        bus.bus_id = static_cast<uint8_t>(obj->attr->pcidev.bus);
        bus.device_id = static_cast<uint8_t>(obj->attr->pcidev.dev);
        bus.function_id = static_cast<uint8_t>(obj->attr->pcidev.func);
        NIXL_INFO << "   " << accelIfaceName(iface) << " accelerator at " << accelBusIdString(bus);
    }
    hwloc_topology_destroy(topo);

    check(vendors_match, "every attributed device carries its vendor's PCI vendor ID");
    if (accelIfaceAvailable(FI_HMEM_ZE) && counts[FI_HMEM_ZE] == 0) {
        NIXL_INFO << "   NOTE: Level Zero is available but enumerated no discrete GPU";
    }
    if (!accelIfaceAvailable(FI_HMEM_ZE)) {
        check(counts[FI_HMEM_ZE] == 0, "no Intel accelerator is reported without Level Zero");
    }
}

/****************************************
 * 7. PCI address rendering
 *****************************************/

void
testBusIdString() {
    NIXL_INFO << "7. Testing PCI address rendering";

    struct fi_pci_attr bus = {};
    bus.bus_id = 0xad;

    // hwloc leaves the domain unpadded and pads the rest. A zero-padded domain here would fail
    // every topology lookup.
    check(accelBusIdString(bus) == "0:ad:00.0",
          "the domain is unpadded and the rest padded, as hwloc writes it");

    bus.domain_id = 0x10;
    bus.bus_id = 0x3a;
    bus.device_id = 0x1f;
    bus.function_id = 3;
    check(accelBusIdString(bus) == "10:3a:1f.3",
          "a non-zero domain and a non-zero function render correctly");

    nixlAccelInfo info;
    info.bus = bus;
    check(accelBusIdString(info).empty(), "an info block with no address renders empty");
    info.bus_kind = nixlAccelBusKind::NIC;
    check(accelBusIdString(info) == "10:3a:1f.3", "an info block with an address renders it");
}

/****************************************
 * 8. Policy
 *****************************************/

void
testPolicy() {
    NIXL_INFO << "8. Testing rail-selection policy";

    check(accelIfaceIsGroupable(FI_HMEM_CUDA), "NVIDIA GPUs join PCIe-proximity grouping");
    check(accelIfaceIsGroupable(FI_HMEM_ZE), "Intel GPUs join PCIe-proximity grouping");
    check(!accelIfaceIsGroupable(FI_HMEM_NEURON), "Neuron devices stay out of grouping");
    check(!accelIfaceIsGroupable(FI_HMEM_SYSTEM), "host memory is not groupable");
    check(!accelIfaceIsGroupable(FI_HMEM_ROCR), "an unhandled interface is not groupable");

    check(accelRequiresStrictAttribution() == accelIfaceAvailable(FI_HMEM_CUDA),
          "strict attribution is required exactly when CUDA is available");
}

/****************************************
 * 9. Context scope
 *****************************************/

void
testScope() {
    NIXL_INFO << "9. Testing context scope";

    {
        // Nothing to bind is success, so callers can construct a scope unconditionally.
        const nixlAccelScope host_guard(FI_HMEM_SYSTEM, 0);
        check(host_guard.ok(), "a host-memory scope reports success");
        const nixlAccelScope ze_guard(FI_HMEM_ZE, 0);
        check(ze_guard.ok(), "a Level Zero scope reports success, having nothing to bind");
    }

    if (!accelIfaceAvailable(FI_HMEM_CUDA)) {
        NIXL_INFO << "   SKIP: CUDA unavailable, cannot test real context push/pop";
        return;
    }

#ifdef CUDA_FOUND
    // Push/pop must leave an application thread exactly as it found it.
    CUcontext before = nullptr;
    const bool have_before = (cuCtxGetCurrent(&before) == CUDA_SUCCESS);

    {
        const nixlAccelScope guard(FI_HMEM_CUDA, 0);
        check(guard.ok(), "a CUDA scope pushes successfully");

        CUcontext inside = nullptr;
        check(cuCtxGetCurrent(&inside) == CUDA_SUCCESS && inside != nullptr,
              "a context is current inside the scope");
    }

    CUcontext after = nullptr;
    const bool have_after = (cuCtxGetCurrent(&after) == CUDA_SUCCESS);
    check(have_before == have_after && before == after,
          "the thread's context is exactly as it was before the scope");

    // Nesting must work too, because registration can be bracketed inside a caller's own scope.
    {
        const nixlAccelScope outer(FI_HMEM_CUDA, 0);
        {
            const nixlAccelScope inner(FI_HMEM_CUDA, 0);
            check(inner.ok(), "a nested CUDA scope pushes successfully");
        }
        CUcontext still = nullptr;
        check(cuCtxGetCurrent(&still) == CUDA_SUCCESS && still != nullptr,
              "the outer scope's context survives the inner scope's pop");
    }

    CUcontext final_ctx = nullptr;
    (void)cuCtxGetCurrent(&final_ctx);
    check(final_ctx == after, "nested scopes unwind to the original binding");
#endif
}

} // namespace

int
main() {
    NIXL_INFO << "=== Testing the libfabric accelerator layer ===";

    testIfaceList();
    testIfaceNames();
    testAvailability();
    testQueryPtr();
    testFillMrAttr();
    testHwlocMatching();
    testBusIdString();
    testPolicy();
    testScope();

    if (failures != 0) {
        NIXL_ERROR << "=== " << failures << " check(s) FAILED ===";
        return 1;
    }
    NIXL_INFO << "=== Test completed successfully! ===";
    return 0;
}
