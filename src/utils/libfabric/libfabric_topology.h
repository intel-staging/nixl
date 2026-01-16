/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025 Amazon.com, Inc. and affiliates.
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
#ifndef NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_TOPOLOGY_H
#define NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_TOPOLOGY_H

#include "libfabric_common.h"
#include "nixl.h"
#include <hwloc.h>
#include <map>

/**
 * @brief Topology discovery and management for libfabric devices
 *
 * Automatically discovers system topology using hwloc and maps GPUs to NICs
 * based on PCIe proximity for optimal performance. Supports EFA, verbs, and other
 * RDMA providers. Falls back to TCP/sockets when RDMA devices are not available.
 */
class nixlLibfabricTopology {
private:
    // GPU to NIC mapping for RDMA providers: GPU 0→[rdmap0s6-rdm,rdmap1s6-rdm], GPU 1→[rdmap2s6-rdm,rdmap3s6-rdm], etc.
    std::map<int, std::vector<std::string>> gpu_to_nics;

    // All available network devices discovered on this system
    std::vector<std::string> all_devices;

    // Network fabric name (efa-direct, efa, tcp, sockets, etc.)
    std::string provider_name;

    // System information
    int num_gpus;  // Total GPUs (NVIDIA + Intel HPU)
    int num_nvidia_gpus;  // NVIDIA GPU count
    int num_intel_hpus;   // Intel Habana HPU count
    int num_numa_nodes;
    int num_devices;

    // Discovery state
    bool topology_discovered;

    // hwloc topology handle
    hwloc_topology_t hwloc_topology;

    // PCIe to Libfabric device mapping
    std::map<std::string, std::string> pcie_to_libfabric_map;
    std::map<std::string, std::string> libfabric_to_pcie_map;

    // Helper methods
    nixl_status_t
    discoverDevices();
    nixl_status_t
    discoverTopology();
    bool
    isRdmaProvider() const;

    // hwloc-based discovery methods
    nixl_status_t
    initHwlocTopology();
    nixl_status_t
    discoverHwlocTopology();
    nixl_status_t
    buildPcieToLibfabricMapping();
    nixl_status_t
    discoverGpusWithHwloc();
    nixl_status_t
    discoverDevicesWithHwloc();
    nixl_status_t
    buildGpuToNicMapping();
    void
    cleanupHwlocTopology();

    // Data structures for NIXL topology-aware grouping algorithm
    struct NicInfo {
        std::string libfabric_name;
        hwloc_obj_t hwloc_node;
        uint16_t domain_id;
        uint8_t bus_id;
        uint8_t device_id;
        uint8_t function_id;
    };

    struct GpuInfo {
        hwloc_obj_t hwloc_node;
        uint16_t domain_id;
        uint8_t bus_id;
        uint8_t device_id;
        uint8_t function_id;
    };

    struct NicGroup {
        std::vector<NicInfo> nics;
        GpuInfo closest_gpu;
        hwloc_obj_t common_ancestor;
        bool has_gpu;
    };

    // NIXL topology-aware grouping algorithm methods
    nixl_status_t
    buildTopologyAwareGrouping();
    nixl_status_t
    buildFallbackMapping();
    nixl_status_t
    groupNicsWithGpus(const std::vector<NicInfo> &discovered_nics,
                      const std::vector<GpuInfo> &discovered_gpus,
                      std::vector<NicGroup> &nic_groups);

    // hwloc helper methods
    std::string
    getPcieAddressFromHwlocObj(hwloc_obj_t obj) const;
    bool
    isIntelHpu(hwloc_obj_t obj) const;
    bool
    isNvidiaGpu(hwloc_obj_t obj) const;
    bool
    isEfaDevice(hwloc_obj_t obj) const;
    bool
    isMellanoxNic(hwloc_obj_t obj) const;

public:
    nixlLibfabricTopology(); // Automatically discovers topology
    ~nixlLibfabricTopology();

    // GPU-based queries (main interface)
    std::vector<std::string>
    getNicsForGpu(int gpu_id) const;

    // System information
    int
    getNumGpus() const {
        return num_gpus;
    }

    int
    getNumNvidiaGpus() const {
        return num_nvidia_gpus;
    }

    int
    getNumIntelHpus() const {
        return num_intel_hpus;
    }

    const std::vector<std::string> &
    getAllDevices() const {
        return all_devices;
    }

    const std::string &
    getProviderName() const {
        return provider_name;
    }

    // Validation
    bool
    isTopologyDiscovered() const {
        return topology_discovered;
    }

    bool
    isValidGpuId(int gpu_id) const;
    bool
    isValidDevice(const std::string &device_name) const;

    // Debug/info
    void
    printTopologyInfo() const;
    std::string
    getTopologyString() const;
};

#endif // NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_TOPOLOGY_H
