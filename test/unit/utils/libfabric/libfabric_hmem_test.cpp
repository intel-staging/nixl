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
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

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

} // namespace

int
main() {
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

    if (failures != 0) {
        NIXL_ERROR << "=== " << failures << " check(s) FAILED ===";
        return 1;
    }
    NIXL_INFO << "=== Test completed successfully! ===";
    return 0;
}
