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
 * Tests for libfabric provider handling: baseProviderName(), the provider table,
 * selectProvider() and selectNetworkDevices(), using synthetic prov_name -> domain maps and
 * synthetic fi_getinfo() lists.
 */

#include "libfabric/libfabric_common.h"
#include "common/nixl_log.h"

#include <algorithm>
#include <deque>
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
 * 2. The provider table
 *****************************************/

void
testProviderTable() {
    NIXL_INFO << "2. Testing the provider table";

    const auto &providers = LibfabricUtils::knownProviders();

    // Preference order is the table order.
    std::vector<std::string> order;
    for (const auto *provider : providers) {
        order.push_back(provider->name);
    }
    NIXL_INFO << "   preference order: [" << join(order) << "]";
    check(order == std::vector<std::string>({"cxi", "efa", "tcp", "sockets"}),
          "providers are in the documented preference order, RDMA before host-stack");

    // Lookup accepts a layered prov_name.
    check(LibfabricUtils::findProviderInfo("efa;ofi_rxd") != nullptr,
          "findProviderInfo() resolves a layered prov_name");
    check(LibfabricUtils::findProviderInfo("efa;ofi_rxd") ==
              LibfabricUtils::findProviderInfo("efa"),
          "the layered and core names resolve to the same entry");
    check(LibfabricUtils::findProviderInfo("psm3") == nullptr,
          "an unlisted provider has no table entry");
    check(LibfabricUtils::findProviderInfo("") == nullptr, "the empty name has no table entry");

    // An unlisted provider gets defaultProviderInfo().
    const auto &fallback = LibfabricUtils::defaultProviderInfo();
    check((fallback.mr_mode & FI_MR_HMEM) != 0,
          "the fallback entry requests FI_MR_HMEM, so device memory still works on an unlisted "
          "provider");

    const auto *cxi = LibfabricUtils::findProviderInfo("cxi");
    const auto *tcp = LibfabricUtils::findProviderInfo("tcp");

    // Per-provider flags.
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
    check(LibfabricUtils::providerNeedsFullTopology("efa") &&
              LibfabricUtils::providerNeedsFullTopology("cxi"),
          "providerNeedsFullTopology() is true for every RDMA provider");
    check(!LibfabricUtils::providerNeedsFullTopology("tcp") &&
              !LibfabricUtils::providerNeedsFullTopology("sockets"),
          "providerNeedsFullTopology() is false for the host-stack providers");
    check(!LibfabricUtils::providerNeedsFullTopology("none"),
          "the no-provider sentinel does not request topology discovery");
}

/****************************************
 * 3. Provider selection over synthetic maps
 *****************************************/

void
testProviderSelection() {
    NIXL_INFO << "3. Testing getAvailableNetworkDevices()'s selection rule";

    auto select = [](const ProviderMap &map) { return LibfabricUtils::selectProvider(map); };

    // efa is preferred over tcp and uses the generic rule.
    const auto efa = select({
        {"efa", {"rdmap16s27-rdm", "rdmap32s27-rdm"}},
        {"tcp", {"lo"}},
    });
    check(efa.first == "efa", "efa outranks tcp");
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
 * 4. Per-provider NIC metadata
 *****************************************/

/** @brief A synthetic fi_getinfo() list; entries stay valid while the list lives. */
class FakeInfoList {
public:
    /** @brief Appends an entry for a NIC on PCI bus @p bus (0:bus:00.0); -1 reports no NIC. */
    void
    add(const std::string &prov_name, const std::string &domain, int bus = -1, size_t speed = 0) {
        Entry &e = entries_.emplace_back();
        e.prov_name = prov_name;
        e.domain = domain;
        e.fabric_attr.prov_name = e.prov_name.data();
        e.domain_attr.name = e.domain.data();
        e.info.fabric_attr = &e.fabric_attr;
        e.info.domain_attr = &e.domain_attr;
        e.info.ep_attr = &e.ep_attr;
        if (bus >= 0) {
            e.bus_attr.bus_type = FI_BUS_PCI;
            e.bus_attr.attr.pci.bus_id = static_cast<uint8_t>(bus);
            e.link_attr.speed = speed;
            e.nic.bus_attr = &e.bus_attr;
            e.nic.link_attr = &e.link_attr;
            e.info.nic = &e.nic;
        }
        if (entries_.size() > 1) {
            entries_[entries_.size() - 2].info.next = &e.info;
        }
    }

    const fi_info *
    head() const {
        return entries_.empty() ? nullptr : &entries_.front().info;
    }

private:
    struct Entry {
        std::string prov_name;
        std::string domain;
        fi_fabric_attr fabric_attr{};
        fi_domain_attr domain_attr{};
        fi_ep_attr ep_attr{};
        fi_bus_attr bus_attr{};
        fi_link_attr link_attr{};
        fid_nic nic{};
        fi_info info{};
    };

    // A deque keeps earlier entries in place, so their next pointers stay valid.
    std::deque<Entry> entries_;
};

void
testDeviceMetadataPerProvider() {
    NIXL_INFO << "4. Testing that NIC metadata comes from the selected provider";

    // An unselected provider lists the same domain name first, with another NIC's metadata.
    FakeInfoList colliding;
    colliding.add("psm3", "rdmap0", 0xaa, 100);
    colliding.add("efa;ofi_rxd", "rdmap0", 0xcc, 50);
    colliding.add("efa", "rdmap0", 0xbb, 200);
    colliding.add("efa", "rdmap0", 0xdd, 400);
    const auto [provider, devices] = LibfabricUtils::selectNetworkDevices(colliding.head());
    check(provider == "efa" && devices.size() == 1, "efa is selected with one rail");
    check(!devices.empty() && devices[0].domain_name == "rdmap0", "the rail keeps its domain");
    check(!devices.empty() && devices[0].pcie_address == "0:bb:00.0",
          "the PCIe address is the selected provider's, not that of an earlier provider "
          "reporting the same domain name");
    check(!devices.empty() && devices[0].link_speed == 200,
          "the link speed is the selected provider's first, not another provider's");

    // The selected provider reports no NIC: another provider's metadata must not fill it in.
    FakeInfoList no_nic;
    no_nic.add("psm3", "rdmap0", 0xaa, 100);
    no_nic.add("efa", "rdmap0");
    const auto [no_nic_provider, no_nic_devices] =
        LibfabricUtils::selectNetworkDevices(no_nic.head());
    check(no_nic_provider == "efa" && no_nic_devices.size() == 1, "efa is selected with one rail");
    check(!no_nic_devices.empty() && no_nic_devices[0].pcie_address.empty() &&
              no_nic_devices[0].link_speed == 0,
          "a domain the selected provider reports no NIC for stays unknown");

    check(LibfabricUtils::selectNetworkDevices(nullptr).first == "none",
          "an empty fi_getinfo() list selects no provider");
}

} // namespace

int
main() {
    NIXL_INFO << "=== Testing Libfabric provider selection ===";

    testBaseProviderName();
    testProviderTable();
    testProviderSelection();
    testDeviceMetadataPerProvider();

    if (failures != 0) {
        NIXL_ERROR << "=== " << failures << " check(s) FAILED ===";
        return 1;
    }
    NIXL_INFO << "=== Test completed successfully! ===";
    return 0;
}
