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
 * Tests for the verbs provider: selectVerbsDomains() over synthetic prov_name -> domain maps, plus
 * a live check that a verbs host resolves PCIe addresses for its NICs (skipped elsewhere).
 */

#include "libfabric/libfabric_common.h"
#include "libfabric/libfabric_topology.h"
#include "common/nixl_log.h"

#include <exception>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

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

using ProviderMap = std::unordered_map<std::string, std::vector<std::string>>;

std::string
join(const std::vector<std::string> &values) {
    std::string out;
    for (const auto &value : values) {
        if (!out.empty()) {
            out += ",";
        }
        out += value;
    }
    return out;
}

/****************************************
 * 1. selectVerbsDomains
 *****************************************/

void
testSelectVerbsDomains() {
    NIXL_INFO << "1. Testing selectVerbsDomains()";

    // FI_EP_RDM as libfabric reports it: several ofi_rxm entries per NIC, plus the ofi_rxd
    // DGRAM domains.
    const ProviderMap real_shape = {
        {"verbs;ofi_rxm",
         {"rocep153s0f0",
          "rocep153s0f0",
          "rocep153s0f0",
          "rocep153s0f0",
          "rocep153s0f1",
          "rocep153s0f1",
          "rocep153s0f1",
          "rocep153s0f1"}},
        {"verbs;ofi_rxd", {"rocep153s0f0-dgram", "rocep153s0f1-dgram"}},
        {"tcp", {"ens1f0np0", "lo"}},
        {"psm3", {"rocep153s0f0", "rocep153s0f1"}},
    };
    const std::vector<std::string> selected = LibfabricUtils::selectVerbsDomains(real_shape);
    NIXL_INFO << "   selected: [" << join(selected) << "]";
    check(selected == std::vector<std::string>({"rocep153s0f0", "rocep153s0f1"}),
          "one rail per NIC, deduplicated, sorted, with the ofi_rxd DGRAM domains dropped");

    // A hyphen in the device name is not an endpoint-variant suffix.
    const ProviderMap hyphenated = {
        {"verbs;ofi_rxm", {"roce-nic0", "roce-nic0-dgram", "roce-nic0-xrc"}},
    };
    const std::vector<std::string> hyphen_selected = LibfabricUtils::selectVerbsDomains(hyphenated);
    NIXL_INFO << "   selected: [" << join(hyphen_selected) << "]";
    check(hyphen_selected == std::vector<std::string>({"roce-nic0"}),
          "a hyphenated device name survives while its -dgram and -xrc variants are dropped");

    check(LibfabricUtils::selectVerbsDomains({{"verbs;ofi_rxd", {"mlx5_0-dgram"}}}).empty(),
          "a host offering verbs only through ofi_rxd selects nothing");
    check(
        LibfabricUtils::selectVerbsDomains({{"efa", {"rdmap16s27-rdm"}}, {"tcp", {"lo"}}}).empty(),
        "no verbs entry selects nothing, and efa's own -rdm domain is left alone");
    check(LibfabricUtils::selectVerbsDomains({}).empty(), "an empty map selects nothing");

    // Domains are sorted, so rail indices do not depend on map order.
    const ProviderMap reversed = {{"verbs;ofi_rxm", {"mlx5_3", "mlx5_1", "mlx5_2", "mlx5_0"}}};
    check(LibfabricUtils::selectVerbsDomains(reversed) ==
              std::vector<std::string>({"mlx5_0", "mlx5_1", "mlx5_2", "mlx5_3"}),
          "domains come back in a deterministic order regardless of discovery order");
}

/****************************************
 * 2. Live topology on a verbs host
 *****************************************/

void
testLiveVerbsTopology() {
    NIXL_INFO << "2. Testing live topology";

    // Skip only when no provider is usable; a topology failure with one found is a failure.
    if (LibfabricUtils::getAvailableNetworkDevices().first == "none") {
        NIXL_INFO << "   no usable libfabric provider on this host, skipping the live checks";
        return;
    }

    std::optional<nixlLibfabricTopology> live;
    try {
        live.emplace();
    }
    catch (const std::exception &e) {
        check(false,
              std::string("topology discovery succeeds when a provider is available: ") + e.what());
        return;
    }
    nixlLibfabricTopology &topology = *live;
    const std::string provider = topology.getProviderName();
    NIXL_INFO << "   provider: " << provider << ", " << topology.getAllDevices().size()
              << " device(s), " << topology.getTotalNicCount() << " NIC(s) with PCIe info";

    if (provider != "verbs") {
        NIXL_INFO << "   not a verbs host, skipping the verbs-specific assertions";
        return;
    }

    // verbs reports FI_BUS_UNKNOWN, so PCIe addresses come from the hwloc fallback.
    check(topology.getTotalNicCount() == topology.getAllDevices().size(),
          "every selected verbs device resolved to a PCIe address");
    check(topology.hasPcieDevices(), "hasPcieDevices() reflects the resolved addresses");

    // Every selected device has a PCIe address.
    for (const auto &device : topology.getAllDevices()) {
        check(!topology.getPcieAddress(device).empty(),
              "selected device " + device + " has a PCIe address");
    }

    for (const auto &device : topology.getAllDevices()) {
        check(device.find("-dgram") == std::string::npos &&
                  device.find("-xrc") == std::string::npos,
              "selected device " + device + " is an RDM-capable domain");
        check(topology.getDeviceNumaNode(device) != nixlLibfabricTopology::INVALID_NUMA_NODE_ID,
              "selected device " + device + " resolved to a NUMA node");
    }
}

} // namespace

int
main() {
    NIXL_INFO << "=== Testing the Libfabric verbs provider ===";

    testSelectVerbsDomains();
    testLiveVerbsTopology();

    if (failures != 0) {
        NIXL_ERROR << "=== " << failures << " check(s) FAILED ===";
        return 1;
    }
    NIXL_INFO << "=== Test completed successfully! ===";
    return 0;
}
