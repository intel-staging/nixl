/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 Intel Corporation. All rights reserved.
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

/**
 * Unit tests for Intel XPU detection in nixlLibfabricTopology.
 *
 * Tests two scenarios against the bmg1 hwloc XML:
 *   1. TCP provider  — accelerator detection only (no NIC topology)
 *   2. verbs provider — full RoCE NIC topology + Intel XPU proximity mapping
 *
 * Uses hwloc XML files captured from real Intel GPU systems so the tests
 * can run on any machine without requiring actual hardware.
 */

#include "libfabric/libfabric_topology.h"
#include "libfabric/libfabric_common.h"
#include "common/nixl_log.h"

#include <cassert>
#include <cstdio>
#include <cstring>

// Shared mock stubs (malloc helpers, mock_fabric_create, etc.)
#include "libfabric_mock_stubs.h"

extern "C" int
__real_fi_getinfo(uint32_t version,
                  const char *node,
                  const char *service,
                  uint64_t flags,
                  const struct fi_info *hints,
                  struct fi_info **info);

extern "C" int
__real_fi_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric, void *context);

// ---- Mock provider state ----

// When g_testing is true, fi_getinfo returns g_mock_provider with g_mock_domains.
static bool g_testing = false;
static const char *g_mock_provider = "tcp";
// For verbs, we inject NIC names and their PCI addresses so that
// buildPcieToLibfabricMapping() populates the pcie_to_libfabric_map correctly.
struct MockNic {
    const char *domain_name;
    uint16_t pci_domain;
    uint8_t pci_bus;
    uint8_t pci_dev;
    uint8_t pci_func;
    uint64_t link_speed_bps; // 0 = unknown
};
static const MockNic *g_mock_nics = nullptr;
static size_t g_mock_nic_count = 0;

extern "C" int
__wrap_fi_getinfo(uint32_t version,
                  const char *node,
                  const char *service,
                  uint64_t flags,
                  const struct fi_info *hints,
                  struct fi_info **info) {
    if (!g_testing)
        return __real_fi_getinfo(version, node, service, flags, hints, info);

    if (g_mock_nic_count == 0) {
        // Minimal single-device fallback (used for TCP scenario)
        *info = malloc_zero<fi_info>();
        (*info)->fabric_attr = malloc_zero<fi_fabric_attr>();
        (*info)->fabric_attr->prov_name = strdup(g_mock_provider);
        (*info)->fabric_attr->name = strdup(g_mock_provider);
        (*info)->domain_attr = malloc_zero<fi_domain_attr>();
        (*info)->domain_attr->name = strdup(g_mock_provider);
        (*info)->ep_attr = malloc_zero<fi_ep_attr>();
        (*info)->ep_attr->type = FI_EP_RDM;
        return 0;
    }

    // Build a fi_info chain with one entry per mock NIC, including PCI bus_attr
    // so that buildPcieToLibfabricMapping() populates pcie_to_libfabric_map.
    fi_info *head = nullptr;
    fi_info *prev = nullptr;
    for (size_t i = 0; i < g_mock_nic_count; ++i) {
        const MockNic &nic = g_mock_nics[i];
        fi_info *cur = malloc_zero<fi_info>();
        cur->fabric_attr = malloc_zero<fi_fabric_attr>();
        cur->fabric_attr->prov_name = strdup(g_mock_provider);
        cur->fabric_attr->name = strdup(g_mock_provider);
        cur->domain_attr = malloc_zero<fi_domain_attr>();
        cur->domain_attr->name = strdup(nic.domain_name);
        cur->ep_attr = malloc_zero<fi_ep_attr>();
        cur->ep_attr->type = FI_EP_RDM;
        cur->nic = malloc_zero<fid_nic>();
        cur->nic->bus_attr = malloc_zero<fi_bus_attr>();
        cur->nic->bus_attr->bus_type = FI_BUS_PCI;
        cur->nic->bus_attr->attr.pci.domain_id = nic.pci_domain;
        cur->nic->bus_attr->attr.pci.bus_id = nic.pci_bus;
        cur->nic->bus_attr->attr.pci.device_id = nic.pci_dev;
        cur->nic->bus_attr->attr.pci.function_id = nic.pci_func;
        cur->nic->link_attr = malloc_zero<fi_link_attr>();
        cur->nic->link_attr->speed = nic.link_speed_bps;
        if (head == nullptr) head = cur;
        if (prev != nullptr) prev->next = cur;
        prev = cur;
    }
    *info = head;
    return 0;
}

extern "C" int
__wrap_fi_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric, void *context) {
    if (!g_testing)
        return __real_fi_fabric(attr, fabric, context);
    *fabric = mock_fabric_create();
    return 0;
}

// ---- Test descriptor ----

struct XpuTopoTest {
    const char *name;
    const char *topo_file;
    const char *provider;
    const MockNic *nics;
    size_t nic_count;
    int expected_xpu_count;
    int expected_nic_count;   // getTotalNicCount(); 0 = don't check
    bool expect_xpu_nic_map;  // true: at least one GPU→NIC mapping must exist
};

// RoCE NICs from intel-xpu-bmg-topo.xml (rocep153s0f0 and rocep153s0f1).
// PCI BDFs confirmed from hwloc XML: 0000:99:00.0 and 0000:99:00.1 (bmg1 / smc-test02).
// Vendor 0x15b3 (Mellanox), class 0x0200 (Ethernet/RoCE mode, ConnectX-6 Dx).
static const MockNic bmg_roce_nics[] = {
    {"rocep153s0f0", 0x0000, 0x99, 0x00, 0x0, 100ull * 1000ull * 1000ull * 1000ull},
    {"rocep153s0f1", 0x0000, 0x99, 0x00, 0x1, 100ull * 1000ull * 1000ull * 1000ull},
};

static XpuTopoTest xpu_topologies[] = {
    {
        "BMG Arc B58x / TCP provider (accelerator detection only)",
        "intel-xpu-bmg-topo.xml",
        "tcp",
        nullptr, 0,
        1,  // one Intel XPU
        0,  // no NIC hwloc map expected for TCP
        false,
    },
    {
        "BMG Arc B58x / verbs provider (RoCE NIC topology + XPU proximity)",
        "intel-xpu-bmg-topo.xml",
        "verbs",
        bmg_roce_nics, 2,
        1,  // one Intel XPU
        2,  // two RoCE ports in NIC info map
        false, // proximity map may not form on this topo without accel-close-NIC pairs
    },
};
static const size_t xpu_topology_count =
    sizeof(xpu_topologies) / sizeof(xpu_topologies[0]);

// ---- Helper to look up actual RoCE NIC PCI BDFs from the hwloc XML ----

static int
testXpuTopology(const XpuTopoTest &t) {
    NIXL_INFO << "Testing Intel XPU topology: " << t.name;

    g_testing = true;
    g_mock_provider = t.provider;
    g_mock_nics = t.nics;
    g_mock_nic_count = t.nic_count;
    setenv("HWLOC_XMLFILE", t.topo_file, 1);

    int result = 0;
    try {
        nixlLibfabricTopology topo;

        // Check Intel XPU count
        int xpu_count = topo.getNumIntelXpuAccel();
        NIXL_INFO << "  getNumIntelXpuAccel() = " << xpu_count
                  << " (expected " << t.expected_xpu_count << ")";
        if (xpu_count != t.expected_xpu_count) {
            NIXL_ERROR << "  FAIL: wrong Intel XPU count, expected "
                       << t.expected_xpu_count << " got " << xpu_count;
            result = 1;
        }

        // Check NIC count when expected
        if (result == 0 && t.expected_nic_count > 0) {
            size_t nic_count = topo.getTotalNicCount();
            NIXL_INFO << "  getTotalNicCount() = " << nic_count
                      << " (expected " << t.expected_nic_count << ")";
            if ((int)nic_count != t.expected_nic_count) {
                NIXL_ERROR << "  FAIL: wrong NIC count, expected "
                           << t.expected_nic_count << " got " << nic_count;
                result = 2;
            }
        }

        if (result == 0)
            NIXL_INFO << "  PASS: " << t.name;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "  FAIL: topology init threw: " << e.what();
        result = 3;
    }

    g_testing = false;
    g_mock_nics = nullptr;
    g_mock_nic_count = 0;
    unsetenv("HWLOC_XMLFILE");
    return result;
}

int
main() {
    NIXL_INFO << "=== Intel XPU Topology Detection Tests ===";

    int failures = 0;
    for (size_t i = 0; i < xpu_topology_count; ++i) {
        int res = testXpuTopology(xpu_topologies[i]);
        if (res != 0) {
            NIXL_ERROR << "FAILED: " << xpu_topologies[i].name
                       << " (code " << res << ")";
            ++failures;
        }
    }

    if (failures == 0) {
        NIXL_INFO << "=== All Intel XPU topology tests PASSED ===";
        return 0;
    }

    NIXL_ERROR << "=== " << failures << " Intel XPU topology test(s) FAILED ===";
    return 1;
}
