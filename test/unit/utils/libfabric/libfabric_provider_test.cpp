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
 * Tests for libfabric provider handling: baseProviderName(), selectVerbsDomains(), the provider
 * table and selectProvider(), using synthetic prov_name -> domain maps; rail pairing by fabric
 * id; and a live check that a verbs host resolves PCIe addresses for its NICs.
 */

#include "libfabric/libfabric_common.h"
#include "libfabric/libfabric_topology.h"
#include "libfabric/libfabric_rail_manager.h"
#include "common/nixl_log.h"

#include <arpa/inet.h>
#include <netinet/in.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
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
 * 1. baseProviderName
 *****************************************/

void
testBaseProviderName() {
    NIXL_INFO << "1. Testing baseProviderName()";

    check(LibfabricUtils::baseProviderName("efa") == "efa", "an unlayered name is returned as is");
    check(LibfabricUtils::baseProviderName("verbs;ofi_rxm") == "verbs",
          "'verbs;ofi_rxm' reduces to 'verbs'");
    check(LibfabricUtils::baseProviderName("verbs;ofi_rxm;ofi_hook_perf") == "verbs",
          "a multiply-layered name reduces to its core provider");
    check(LibfabricUtils::baseProviderName("") == "", "the empty name is handled");
    check(LibfabricUtils::baseProviderName(";ofi_rxm") == "",
          "a leading separator yields an empty core name rather than reading past it");
    // verbs reports its RDM endpoint as the layered "verbs;ofi_rxm".
    check(LibfabricUtils::baseProviderName("verbs;ofi_rxm") != "verbs;ofi_rxm",
          "the layered name is not equal to its core name");
}

/****************************************
 * 2. selectVerbsDomains
 *****************************************/

void
testSelectVerbsDomains() {
    NIXL_INFO << "2. Testing selectVerbsDomains()";

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
 * 3. The provider table
 *****************************************/

void
testProviderTable() {
    NIXL_INFO << "3. Testing the provider table";

    const auto &providers = LibfabricUtils::knownProviders();

    // Preference order is the table order.
    std::vector<std::string> order;
    for (const auto *provider : providers) {
        order.push_back(provider->name);
    }
    NIXL_INFO << "   preference order: [" << join(order) << "]";
    check(order == std::vector<std::string>({"cxi", "efa", "verbs", "tcp", "sockets"}),
          "providers are in the documented preference order, RDMA before host-stack");

    // Lookup accepts a layered prov_name.
    check(LibfabricUtils::findProviderInfo("verbs;ofi_rxm") != nullptr,
          "findProviderInfo() resolves a layered prov_name");
    check(LibfabricUtils::findProviderInfo("verbs;ofi_rxm") ==
              LibfabricUtils::findProviderInfo("verbs"),
          "the layered and core names resolve to the same entry");
    check(LibfabricUtils::findProviderInfo("psm3") == nullptr,
          "an unlisted provider has no table entry");
    check(LibfabricUtils::findProviderInfo("") == nullptr, "the empty name has no table entry");

    // An unlisted provider gets defaultProviderInfo().
    const auto &fallback = LibfabricUtils::defaultProviderInfo();
    check((fallback.mr_mode & FI_MR_HMEM) != 0,
          "the fallback entry requests FI_MR_HMEM, so device memory still works on an unlisted "
          "provider");
    check(!fallback.needs_prov_name_hint,
          "the fallback entry does not pin hints to one provider name");

    const auto *verbs = LibfabricUtils::findProviderInfo("verbs");
    const auto *cxi = LibfabricUtils::findProviderInfo("cxi");
    const auto *tcp = LibfabricUtils::findProviderInfo("tcp");

    // Per-provider flags.
    check(verbs != nullptr && verbs->needs_prov_name_hint,
          "verbs names itself in rail hints, so psm3 cannot claim the same domain");
    check(verbs != nullptr && verbs->needs_all_rail_progress,
          "verbs progresses every rail, so the passive side accepts lazy connections");
    check(cxi != nullptr && !cxi->needs_all_rail_progress,
          "a connectionless RDM provider keeps the cheaper active-rails-only progress");
    check(cxi != nullptr && (cxi->extra_caps & FI_RMA_EVENT) != 0, "cxi requests FI_RMA_EVENT");
    check(cxi != nullptr && (cxi->mr_mode & FI_MR_ENDPOINT) != 0,
          "cxi requests FI_MR_ENDPOINT in its mr_mode");
    check(tcp != nullptr && (tcp->mr_mode & (FI_MR_PROV_KEY | FI_MR_VIRT_ADDR)) == 0,
          "tcp asks for neither FI_MR_PROV_KEY nor FI_MR_VIRT_ADDR, which it does not support");

    // Only the host-stack providers use a single rail.
    for (const auto *provider : providers) {
        const bool is_host_stack =
            (std::string(provider->name) == "tcp" || std::string(provider->name) == "sockets");
        check(provider->single_domain == is_host_stack,
              std::string(provider->name) +
                  (is_host_stack ? " uses a single rail" : " uses one rail per NIC"));
        check(provider->needs_full_topology == !is_host_stack,
              std::string(provider->name) +
                  (is_host_stack ? " skips hwloc grouping" : " does PCIe-proximity grouping"));
    }

    // The flag helpers used by the rest of the plugin.
    check(LibfabricUtils::providerNeedsAllRailProgress("verbs;ofi_rxm"),
          "providerNeedsAllRailProgress() accepts the layered name it will really be given");
    check(!LibfabricUtils::providerNeedsAllRailProgress("efa"),
          "providerNeedsAllRailProgress() is false for efa");
    check(!LibfabricUtils::providerNeedsAllRailProgress("psm3"),
          "an unlisted provider does not get all-rail progress");
    check(LibfabricUtils::providerNeedsFullTopology("efa") &&
              LibfabricUtils::providerNeedsFullTopology("cxi") &&
              LibfabricUtils::providerNeedsFullTopology("verbs"),
          "providerNeedsFullTopology() is true for every RDMA provider");
    check(!LibfabricUtils::providerNeedsFullTopology("tcp") &&
              !LibfabricUtils::providerNeedsFullTopology("sockets"),
          "providerNeedsFullTopology() is false for the host-stack providers");
    check(!LibfabricUtils::providerNeedsFullTopology("none"),
          "the no-provider sentinel does not request topology discovery");
}

/****************************************
 * 4. Provider selection over synthetic maps
 *****************************************/

void
testProviderSelection() {
    NIXL_INFO << "4. Testing getAvailableNetworkDevices()'s selection rule";

    auto select = [](const ProviderMap &map) { return LibfabricUtils::selectProvider(map); };

    // verbs is preferred over tcp.
    const auto roce = select({
        {"verbs;ofi_rxm", {"rocep153s0f0", "rocep153s0f1"}},
        {"verbs;ofi_rxd", {"rocep153s0f0-dgram"}},
        {"tcp", {"ens1f0np0", "lo"}},
        {"psm3", {"rocep153s0f0"}},
    });
    check(roce.first == "verbs", "verbs is preferred over tcp on a RoCE host");
    check(roce.second == std::vector<std::string>({"rocep153s0f0", "rocep153s0f1"}),
          "the verbs hook's filtered domain list is the one used");

    // efa is preferred over verbs and uses the generic rule.
    const auto efa = select({
        {"efa", {"rdmap16s27-rdm", "rdmap32s27-rdm"}},
        {"verbs;ofi_rxm", {"mlx5_0"}},
        {"tcp", {"lo"}},
    });
    check(efa.first == "efa", "efa outranks verbs");
    check(
        efa.second == std::vector<std::string>({"rdmap16s27-rdm", "rdmap32s27-rdm"}),
        "the generic rule preserves fi_getinfo order, so existing efa rail numbering is unchanged");

    check(select({{"cxi", {"cxi0"}}, {"efa", {"rdmap16s27-rdm"}}}).first == "cxi",
          "cxi outranks efa");

    // The generic rule deduplicates domains.
    const auto dupes = select({{"efa", {"rdmap16s27-rdm", "rdmap16s27-rdm", "rdmap32s27-rdm"}}});
    check(dupes.second == std::vector<std::string>({"rdmap16s27-rdm", "rdmap32s27-rdm"}),
          "the generic rule deduplicates repeated domains");

    // ...and ignores layered variants of the provider.
    const auto layered = select({
        {"efa;ofi_rxd", {"rdmap16s27-dgrm", "rdmap32s27-dgrm"}},
        {"efa", {"rdmap16s27-rdm", "rdmap32s27-rdm"}},
    });
    check(layered.first == "efa", "efa is selected when it also has layered variants");
    check(layered.second == std::vector<std::string>({"rdmap16s27-rdm", "rdmap32s27-rdm"}),
          "the generic rule ignores layered variants such as efa;ofi_rxd");

    // tcp and sockets collapse to one rail however many domains they report.
    const auto tcp = select({{"tcp", {"ens1f0np0", "lo", "docker0"}}});
    check(tcp.first == "tcp" && tcp.second.size() == 1,
          "tcp collapses to a single rail regardless of how many domains it offers");
    check(select({{"sockets", {"a", "b"}}}).second.size() == 1, "sockets collapses too");

    check(select({}).first == "none", "an empty map selects no provider");
    check(select({{"psm3", {"rocep153s0f0"}}}).first == "none",
          "a host offering only an unlisted provider selects nothing rather than mis-claiming it");
}

/****************************************
 * 5. Live topology on a verbs host
 *****************************************/

void
testLiveVerbsTopology() {
    NIXL_INFO << "5. Testing live topology";

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

void
testFabricId() {
    NIXL_INFO << "Testing fabric ids";

    auto ep_name = [](const char *ip) {
        struct sockaddr_in sin = {};
        sin.sin_family = AF_INET;
        inet_pton(AF_INET, ip, &sin.sin_addr);
        std::array<char, sizeof(sin)> buf;
        memcpy(buf.data(), &sin, sizeof(sin));
        return buf;
    };
    const auto a = ep_name("192.168.10.6");
    auto local = LibfabricUtils::fabricIdFromEpName(FI_SOCKADDR_IN, a.data(), a.size());
    check(local.valid, "an IPv4 endpoint name gives a valid fabric id");
    check(!LibfabricUtils::fabricIdFromEpName(FI_ADDR_EFA, a.data(), a.size()).valid,
          "an unreadable address format gives an invalid id");
    check(!LibfabricUtils::fabricIdFromEpName(FI_SOCKADDR_IN, a.data(), 4).valid,
          "a short endpoint name gives an invalid id");

    const auto same = ep_name("192.168.10.16");
    const auto other = ep_name("192.168.20.6");
    const auto peer_same =
        LibfabricUtils::fabricIdFromEpName(FI_SOCKADDR_IN, same.data(), same.size());
    const auto peer_other =
        LibfabricUtils::fabricIdFromEpName(FI_SOCKADDR_IN, other.data(), other.size());

    check(!LibfabricUtils::sameFabric(local, peer_same),
          "without a netmask only an exact address matches");
    inet_pton(AF_INET, "255.255.255.0", &local.mask);
    check(LibfabricUtils::sameFabric(local, peer_same), "an address on the subnet matches");
    check(!LibfabricUtils::sameFabric(local, peer_other), "an address off the subnet does not");
    check(!LibfabricUtils::sameFabric(local, LibfabricUtils::nixlLibfabricFabricId{}),
          "an invalid id never matches");
}

void
testFallbackPeerRails() {
    NIXL_INFO << "Testing fallback peer rail assignment";
    using V = std::vector<size_t>;
    const size_t none = NIXL_LF_NO_PEER_RAIL;
    const auto assign = [](const V &selected,
                           const V &registered,
                           const V &peering,
                           const nixlLibfabricRailUnreachable &unreachable = {}) {
        return nixlLibfabricRailManager::assignFallbackPeerRails(
            selected, registered, peering, unreachable);
    };

    // Without fabric ids a registered same-id peer rail wins (a rail-optimised fabric needs it)
    check(assign({2, 3}, {0, 1, 2, 3}, {}) == V({2, 3}),
          "without fabric ids, a registered same-id peer rail is preferred");

    // Policy order does not change the pairing, and it stays one-to-one
    check(assign({2, 0}, {0, 1}, {}) == V({1, 0}),
          "without a same-id peer rail, rails pair by rank in rail-id order, one-to-one");
    check(assign({1, 0}, {0, 1}, {}) == V({1, 0}), "policy order does not change the pairing");

    // A fabric match claims its peer rail; an unmatched rail takes another
    check(assign({0, 1}, {0, 1}, {1, none}) == V({1, 0}),
          "an unmatched rail takes an unclaimed peer rail, not one a fabric match holds");

    // Matched peer rail unregistered: the same-id rail is on another segment, so it is skipped
    check(assign({0, 1}, {0, 2}, {2, 3}) == V({2, 0}),
          "a rail whose matched peer is unregistered skips the same-id rail");

    // More local rails than registered peer rails: sharing only once all are claimed
    check(assign({0, 1, 2}, {0}, {}) == V({0, 0, 0}),
          "with one registered peer rail, every local rail shares it");
    check(assign({0, 1, 2, 3}, {0, 1}, {}) == V({0, 1, 0, 1}),
          "sharing only starts once every registered peer rail is claimed");

    check(assign({0, 1}, {}, {}) == V({none, none}),
          "no registered peer rail leaves the rails unassigned rather than indexing an empty list");

    // Rails cabled across, buffer on peer rail 0 only: local rail 0 cannot reach it
    const nixlLibfabricRailUnreachable crossed = {{true, false}, {false, true}};
    check(assign({0, 1}, {0}, {1, 0}, crossed) == V({none, 0}),
          "a rail that reaches no registered peer rail is left unassigned, not given another's");
    check(assign({0, 1}, {0}, {1, 0}) == V({0, 0}),
          "without reachability information, sharing is unchanged");

    // One subnet, buffers on disjoint rail sets: every pair is reachable
    const nixlLibfabricRailUnreachable flat(4, std::vector<bool>(4, false));
    check(assign({0, 1}, {2, 3}, {0, 1, 2, 3}, flat) == V({2, 3}),
          "on a single segment, an unregistered matched peer rail does not strand the rail");

    // Sharing skips unreachable peer rails: local rail 2 reaches only peer rail 1
    const nixlLibfabricRailUnreachable partial = {{false, false}, {false, false}, {true, false}};
    check(assign({0, 1, 2}, {0, 1}, {}, partial) == V({0, 1, 1}),
          "sharing picks among the peer rails a rail can reach");

    // Both local rails on segment A, peer rail 1 on B: local rail 1 shares peer rail 0
    const nixlLibfabricRailUnreachable one_segment = {{false, true}, {false, true}};
    check(assign({0, 1}, {0, 1}, {0, none}, one_segment) == V({0, 0}),
          "same-id pairing skips a peer rail the rail is known not to reach");

    std::vector<size_t> storage;
    const auto usable = [&](const V &selected,
                            const V &registered,
                            const nixlLibfabricRailUnreachable &unreachable) {
        return nixlLibfabricRailManager::usableRails(selected, registered, unreachable, storage);
    };
    check(usable({0, 1}, {0}, crossed) == V({1}),
          "usableRails drops a rail that reaches no registered peer rail");
    check(usable({1, 0}, {0, 1}, crossed) == V({1, 0}),
          "usableRails keeps every rail, in policy order, when each reaches one");
    check(usable({0}, {0}, crossed).empty(), "usableRails is empty when no rail is usable");
    check(usable({0, 1}, {0}, {}) == V({0, 1}),
          "usableRails keeps every rail without reachability information");
    const V selected = {0, 1};
    check(&nixlLibfabricRailManager::usableRails(selected, {0, 1}, crossed, storage) == &selected,
          "usableRails returns the caller's list, not a copy, when nothing is dropped");
    check(&nixlLibfabricRailManager::usableRails(selected, {0}, {}, storage) == &selected,
          "usableRails returns the caller's list at once when no pair is marked unreachable");
}

int
main() {
    NIXL_INFO << "=== Testing Libfabric verbs provider selection ===";

    testBaseProviderName();
    testSelectVerbsDomains();
    testProviderTable();
    testProviderSelection();
    testFabricId();
    testFallbackPeerRails();
    testLiveVerbsTopology();

    if (failures != 0) {
        NIXL_ERROR << "=== " << failures << " check(s) FAILED ===";
        return 1;
    }
    NIXL_INFO << "=== Test completed successfully! ===";
    return 0;
}
