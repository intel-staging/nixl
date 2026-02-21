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
 * Uses hwloc XML topology files to validate isIntelXpuAccel() detection
 * and getNumIntelXpuAccel() count on Intel BMG/Arc/Max/Flex systems.
 * Tests are driven by static topology descriptors; each entry names the
 * hwloc XML file and the expected Intel XPU count.
 */

#include "libfabric/libfabric_topology.h"
#include "libfabric/libfabric_common.h"
#include "common/nixl_log.h"

#include <cassert>
#include <cstdio>
#include <cstring>

// mock fi_getinfo / fi_fabric to avoid requiring real hardware
extern "C" int
__wrap_fi_getinfo(uint32_t version,
                  const char *node,
                  const char *service,
                  uint64_t flags,
                  const struct fi_info *hints,
                  struct fi_info **info);

extern "C" int
__wrap_fi_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric, void *context);

// shared stubs (same pattern as libfabric_topology_test.cpp)
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

static bool g_testing = false;

extern "C" int
__wrap_fi_getinfo(uint32_t version,
                  const char *node,
                  const char *service,
                  uint64_t flags,
                  const struct fi_info *hints,
                  struct fi_info **info) {
    if (!g_testing)
        return __real_fi_getinfo(version, node, service, flags, hints, info);

    // Return a minimal fi_info with a TCP provider so topology init succeeds
    // without requiring real EFA hardware.
    *info = malloc_zero<fi_info>();
    (*info)->fabric_attr = malloc_zero<fi_fabric_attr>();
    (*info)->fabric_attr->prov_name = strdup("tcp");
    (*info)->fabric_attr->name = strdup("tcp");
    (*info)->domain_attr = malloc_zero<fi_domain_attr>();
    (*info)->domain_attr->name = strdup("tcp");
    (*info)->ep_attr = malloc_zero<fi_ep_attr>();
    (*info)->ep_attr->type = FI_EP_RDM;
    return 0;
}

extern "C" int
__wrap_fi_fabric(struct fi_fabric_attr *attr, struct fid_fabric **fabric, void *context) {
    if (!g_testing)
        return __real_fi_fabric(attr, fabric, context);
    *fabric = mock_fabric_create();
    return 0;
}

struct XpuTopoTest {
    const char *name;       // human-readable test name
    const char *topo_file;  // hwloc XML filename (relative, loaded via HWLOC_XMLFILE)
    int expected_xpu_count; // expected getNumIntelXpuAccel()
    int expected_numa_nodes; // expected NUMA node count (0 = don't check)
};

static XpuTopoTest xpu_topologies[] = {
    {
        "BMG Arc B58x (bmg1 / smc-test02)",
        "intel-xpu-bmg-topo.xml",
        1, // one BMG GPU at 0000:ad:00.0
        0, // don't assert NUMA count for this topology
    },
};
static const size_t xpu_topology_count =
    sizeof(xpu_topologies) / sizeof(xpu_topologies[0]);

static int
testXpuTopology(const XpuTopoTest &t) {
    NIXL_INFO << "Testing Intel XPU topology: " << t.name;

    g_testing = true;
    setenv("HWLOC_XMLFILE", t.topo_file, 1);

    int result = 0;
    try {
        nixlLibfabricTopology topo;

        int xpu_count = topo.getNumIntelXpuAccel();
        NIXL_INFO << "  getNumIntelXpuAccel() = " << xpu_count
                  << " (expected " << t.expected_xpu_count << ")";

        if (xpu_count != t.expected_xpu_count) {
            NIXL_ERROR << "  FAIL: wrong Intel XPU count, expected "
                       << t.expected_xpu_count << " got " << xpu_count;
            result = 1;
        }

        if (t.expected_numa_nodes > 0) {
            // Validate via device iteration as a proxy for NUMA node count
            // (topology doesn't expose raw NUMA count directly)
            NIXL_DEBUG << "  Skipping direct NUMA count check (no public accessor)";
        }

        if (result == 0)
            NIXL_INFO << "  PASS: " << t.name;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "  FAIL: topology init threw: " << e.what();
        result = 2;
    }

    g_testing = false;
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
