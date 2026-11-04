/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * Tests for libfabric provider handling: baseProviderName(), the provider table, the rail hint
 * and registration-key helpers, selectProvider() and selectNetworkDevices(), using synthetic
 * prov_name -> domain maps and synthetic fi_getinfo() lists.
 */

#include "libfabric/libfabric_common.h"
#include "common/nixl_log.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>

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

using ProviderMap = LibfabricUtils::nixlProviderDeviceMap;

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

    // The exact settings each provider had before the table, so the move changes none of them.
    struct Expected {
        const char *name;
        uint64_t extra_caps;
        uint64_t mr_mode;
        size_t mr_key_size;
        uint64_t mr_access;
    };

    constexpr uint64_t rdma_mode =
        FI_MR_LOCAL | FI_MR_HMEM | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY;
    constexpr uint64_t basic_access = FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE;
    const Expected expected[] = {
        {"cxi",
         FI_RMA_EVENT,
         rdma_mode | FI_MR_ENDPOINT,
         0,
         FI_REMOTE_WRITE | FI_REMOTE_READ | FI_RMA_EVENT},
        {"efa", 0, rdma_mode, 2, FI_REMOTE_WRITE | FI_REMOTE_READ},
        {"tcp", 0, FI_MR_LOCAL | FI_MR_ALLOCATED, 0, basic_access},
        {"sockets", 0, FI_MR_LOCAL | FI_MR_ALLOCATED, 0, basic_access},
    };
    for (const Expected &e : expected) {
        const auto *info = LibfabricUtils::findProviderInfo(e.name);
        const std::string name = e.name;
        check(info != nullptr, name + " has a table entry");
        if (info == nullptr) {
            continue;
        }
        check(info->extra_caps == e.extra_caps, name + " keeps its extra caps");
        check(info->mr_mode == e.mr_mode, name + " keeps its exact mr_mode");
        check(info->mr_key_size == e.mr_key_size, name + " keeps its mr_key_size");
        check(info->mr_access == e.mr_access, name + " keeps its registration access flags");
    }
    check(&fallback == LibfabricUtils::findProviderInfo("efa"),
          "the fallback entry is efa's, the settings an unlisted provider had before");

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
    bool any_notifies_after_completion = false;
    for (const auto *provider : providers) {
        any_notifies_after_completion |=
            LibfabricUtils::providerNotifiesAfterCompletion(provider->name);
    }
    check(!any_notifies_after_completion,
          "every listed provider has the target count a WRITE's writes");
    check(!LibfabricUtils::providerNotifiesAfterCompletion("psm3"),
          "an unlisted provider has the target count a WRITE's writes");
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
 * 3. Rail hints and registration keys
 *****************************************/

void
testRailHelpers() {
    NIXL_INFO << "3. Testing applyProviderRailHints() and providerRequestsMrKey()";

    constexpr uint64_t base_caps = FI_MSG | FI_RMA | FI_LOCAL_COMM | FI_REMOTE_COMM | FI_SOURCE;
    for (const auto *provider : LibfabricUtils::knownProviders()) {
        const std::string name = provider->name;
        fi_info *hints = fi_allocinfo();
        check(hints != nullptr, "fi_allocinfo() succeeds for " + name);
        if (hints == nullptr) {
            continue;
        }

        hints->caps = base_caps | FI_HMEM;
        LibfabricUtils::applyProviderRailHints(*provider, hints);
        check(hints->caps == (base_caps | FI_HMEM | provider->extra_caps),
              name + " rail hints add exactly its extra caps");
        // fi_domain_attr::mr_mode is an int.
        check(static_cast<uint64_t>(hints->domain_attr->mr_mode) == provider->mr_mode,
              name + " rail hints carry its mr_mode");
        check(hints->domain_attr->mr_key_size == provider->mr_key_size,
              name + " rail hints carry its mr_key_size");

        // The retry without FI_HMEM resets caps and applies the entry again.
        hints->caps = base_caps;
        LibfabricUtils::applyProviderRailHints(*provider, hints);
        check(hints->caps == (base_caps | provider->extra_caps),
              name + " keeps its extra caps on the retry without FI_HMEM");

        fi_freeinfo(hints);
    }

    // A registration requests its own key exactly when the provider does not choose one.
    for (const char *name : {"tcp", "sockets"}) {
        const auto *info = LibfabricUtils::findProviderInfo(name);
        check(info != nullptr && LibfabricUtils::providerRequestsMrKey(*info),
              std::string(name) + " registrations request a unique key (no FI_MR_PROV_KEY)");
    }
    for (const char *name : {"cxi", "efa"}) {
        const auto *info = LibfabricUtils::findProviderInfo(name);
        check(info != nullptr && !LibfabricUtils::providerRequestsMrKey(*info),
              std::string(name) + " registrations leave the key to the provider");
    }
    check(!LibfabricUtils::providerRequestsMrKey(LibfabricUtils::defaultProviderInfo()),
          "an unlisted provider's registrations leave the key to the provider, as efa's do");
}

/****************************************
 * 4. Provider selection over synthetic maps
 *****************************************/

void
testProviderSelection() {
    NIXL_INFO << "4. Testing selectProvider()";

    auto select = [](const ProviderMap &map) { return LibfabricUtils::selectProvider(map); };

    // efa is preferred over tcp and uses the generic rule.
    const auto efa = select({
        {"efa", {"rdmap16s27-rdm", "rdmap32s27-rdm"}},
        {"tcp", {"lo"}},
    });
    check(efa.first == "efa", "efa outranks tcp");
    check(efa.second == std::vector<std::string>({"rdmap16s27-rdm", "rdmap32s27-rdm"}),
          "selection preserves fi_getinfo order, so existing efa rail numbering is unchanged");

    check(select({{"cxi", {"cxi0"}}, {"efa", {"rdmap16s27-rdm"}}}).first == "cxi",
          "cxi outranks efa");

    // A domain listed twice (e.g. once per fabric) gets one rail, for cxi as well as efa.
    const auto dupes = select({{"efa", {"rdmap16s27-rdm", "rdmap16s27-rdm", "rdmap32s27-rdm"}}});
    check(dupes.second == std::vector<std::string>({"rdmap16s27-rdm", "rdmap32s27-rdm"}),
          "repeated efa domains are deduplicated, keeping first-seen order");
    const auto cxi_dupes = select({{"cxi", {"cxi0", "cxi1", "cxi0"}}});
    check(cxi_dupes.second == std::vector<std::string>({"cxi0", "cxi1"}),
          "repeated cxi domains are deduplicated, keeping first-seen order");

    // Layered variants of the provider are ignored.
    const auto layered = select({
        {"efa;ofi_rxd", {"rdmap16s27-dgrm", "rdmap32s27-dgrm"}},
        {"efa", {"rdmap16s27-rdm", "rdmap32s27-rdm"}},
    });
    check(layered.first == "efa", "efa is selected when it also has layered variants");
    check(layered.second == std::vector<std::string>({"rdmap16s27-rdm", "rdmap32s27-rdm"}),
          "selection ignores layered variants such as efa;ofi_rxd");

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
 * 5. Per-provider NIC metadata
 *****************************************/

/** @brief A synthetic fi_getinfo() list; entries stay valid while the list lives. */
class FakeInfoList {
public:
    FakeInfoList() = default;
    // Entries point into themselves and into each other, so a copy would point into the original.
    FakeInfoList(const FakeInfoList &) = delete;
    FakeInfoList &
    operator=(const FakeInfoList &) = delete;
    FakeInfoList(FakeInfoList &&) = delete;
    FakeInfoList &
    operator=(FakeInfoList &&) = delete;

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
        link();
    }

    /** @brief Appends an entry with no prov_name, no domain name and no ep_attr. */
    void
    addWithoutNames() {
        Entry &e = entries_.emplace_back();
        e.info.fabric_attr = &e.fabric_attr;
        e.info.domain_attr = &e.domain_attr;
        link();
    }

    const fi_info *
    head() const {
        return entries_.empty() ? nullptr : &entries_.front().info;
    }

private:
    void
    link() {
        if (entries_.size() > 1) {
            entries_[entries_.size() - 2].info.next = &entries_.back().info;
        }
    }

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
    NIXL_INFO << "5. Testing that NIC metadata comes from the selected provider";

    // An unselected provider, and another stack of the selected one, list the same domain name
    // first, with other NICs' metadata.
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
          "the link speed is the first one efa lists for the domain (200, not its later 400), "
          "never another provider's (100, 50)");

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

    // Entries without names are skipped rather than dereferenced.
    FakeInfoList nameless;
    nameless.addWithoutNames();
    nameless.add("efa", "rdmap0", 0xbb, 200);
    nameless.addWithoutNames();
    const auto [nameless_provider, nameless_devices] =
        LibfabricUtils::selectNetworkDevices(nameless.head());
    check(nameless_provider == "efa" && nameless_devices.size() == 1 &&
              nameless_devices[0].domain_name == "rdmap0",
          "entries with no prov_name or domain name are skipped");

    // tcp keeps one rail, its first domain, with that domain's metadata.
    FakeInfoList tcp;
    tcp.add("tcp", "ens1f0np0", 0x17, 100);
    tcp.add("tcp", "lo");
    tcp.add("tcp", "docker0");
    const auto [tcp_provider, tcp_devices] = LibfabricUtils::selectNetworkDevices(tcp.head());
    check(tcp_provider == "tcp" && tcp_devices.size() == 1, "tcp is selected with one rail");
    check(!tcp_devices.empty() && tcp_devices[0].domain_name == "ens1f0np0" &&
              tcp_devices[0].pcie_address == "0:17:00.0" && tcp_devices[0].link_speed == 100,
          "the tcp rail is the first domain, with its own PCIe address and link speed");

    check(LibfabricUtils::selectNetworkDevices(nullptr).first == "none",
          "an empty fi_getinfo() list selects no provider");
}

} // namespace

int
main() {
    NIXL_INFO << "=== Testing Libfabric provider selection ===";

    testBaseProviderName();
    testProviderTable();
    testRailHelpers();
    testProviderSelection();
    testDeviceMetadataPerProvider();

    if (failures != 0) {
        NIXL_ERROR << "=== " << failures << " check(s) FAILED ===";
        return 1;
    }
    NIXL_INFO << "=== Test completed successfully! ===";
    return 0;
}
