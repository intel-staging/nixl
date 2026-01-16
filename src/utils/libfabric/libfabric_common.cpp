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

#include "libfabric_common.h"
#include "common/nixl_log.h"

#include <iomanip>
#include <sstream>
#include <atomic>
#include <cstring>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>

namespace LibfabricUtils {

// Provider-specific configurations
static const ProviderConfig PROVIDER_CONFIGS[] = {
    {"efa",
     FI_MSG | FI_RMA | FI_LOCAL_COMM | FI_REMOTE_COMM,
     FI_CONTEXT | FI_CONTEXT2,
     0, // let provider choose
     FI_RM_UNSPEC,
     FI_THREAD_SAFE},
    {"verbs", // Matches both "verbs" and "verbs;ofi_rxm"
     FI_MSG | FI_RMA | FI_READ | FI_WRITE | FI_RECV | FI_SEND | FI_REMOTE_READ | FI_REMOTE_WRITE |
         FI_MULTI_RECV | FI_LOCAL_COMM | FI_REMOTE_COMM | FI_HMEM,
     0, // no mode flags required
     FI_MR_LOCAL | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_HMEM,
     FI_RM_ENABLED,
     FI_THREAD_SAFE},
    {"tcp",
     FI_MSG | FI_RMA | FI_LOCAL_COMM | FI_REMOTE_COMM,
     FI_CONTEXT | FI_CONTEXT2,
     0, // basic MR mode, overridden in rail.cpp
     FI_RM_UNSPEC,
     FI_THREAD_UNSPEC},
    {
        "sockets",
        FI_MSG | FI_RMA | FI_LOCAL_COMM | FI_REMOTE_COMM,
        0,
        0, // let provider choose
        FI_RM_UNSPEC,
        FI_THREAD_UNSPEC // default threading
    }};

static const size_t NUM_PROVIDER_CONFIGS = sizeof(PROVIDER_CONFIGS) / sizeof(PROVIDER_CONFIGS[0]);

void
configureHintsForProvider(struct fi_info *hints, const std::string &provider_name) {
    const ProviderConfig *config = nullptr;

    // Find matching config
    // Match order: 1) exact match, 2) prefix match for composite providers (e.g., "verbs;ofi_rxm")
    for (size_t i = 0; i < NUM_PROVIDER_CONFIGS; ++i) {
        const std::string &config_name = PROVIDER_CONFIGS[i].name;

        // Exact match
        if (provider_name == config_name) {
            config = &PROVIDER_CONFIGS[i];
            break;
        }

        // Composite provider match (e.g., "verbs;ofi_rxm" matches "verbs")
        // Check if provider_name starts with config_name followed by ";"
        if (provider_name.rfind(config_name + ";", 0) == 0) {
            config = &PROVIDER_CONFIGS[i];
            break;
        }
    }

    if (!config) {
        // Default configuration
        NIXL_DEBUG << "No specific config for provider '" << provider_name << "', using defaults";
        hints->caps = FI_MSG | FI_RMA | FI_LOCAL_COMM | FI_REMOTE_COMM;
        hints->mode = 0;
        hints->ep_attr->type = FI_EP_RDM;
        return;
    }

    // Apply provider-specific configuration
    hints->caps = config->caps;
    hints->mode = config->mode;
    hints->ep_attr->type = FI_EP_RDM;

    if (config->resource_mgmt != FI_RM_UNSPEC) {
        hints->domain_attr->resource_mgmt = config->resource_mgmt;
    }

    if (config->mr_mode != 0) {
        hints->domain_attr->mr_mode = config->mr_mode;
    }

    if (config->threading != FI_THREAD_UNSPEC) {
        hints->domain_attr->threading = config->threading;
    }
}

std::pair<std::string, std::vector<std::string>>
getAvailableNetworkDevices() {
    std::vector<std::string> all_devices;
    std::string provider_name;

    std::unordered_map<std::string, std::vector<std::string>> provider_device_map;
    struct fi_info *hints, *info;
    hints = fi_allocinfo();
    if (!hints) {
        NIXL_ERROR << "Failed to allocate fi_info";
        return {"none", {}};
    }

    // Check if FI_PROVIDER environment variable is set
    const char *env_provider = getenv("FI_PROVIDER");
    std::string provider = env_provider && env_provider[0] != '\0' ? env_provider : "";

    if (!provider.empty()) {
        hints->fabric_attr->prov_name = strdup(env_provider);
        NIXL_INFO << "Using provider from FI_PROVIDER environment: " << env_provider;
        // Configure hints based on provider
        configureHintsForProvider(hints, provider);
    } else {
        // Auto-detect: start with default configuration
        configureHintsForProvider(hints, "");
    }

    // Use FI_VERSION(1, 18) for DMABUF and HMEM support
    int ret = fi_getinfo(FI_VERSION(1, 18), NULL, NULL, 0, hints, &info);
    if (ret) {
        NIXL_ERROR << "fi_getinfo failed: " << fi_strerror(-ret);
        fi_freeinfo(hints);
        return {"none", {}};
    }

    // Process devices for this provider
    for (struct fi_info *cur = info; cur; cur = cur->next) {
        if (cur->domain_attr && cur->domain_attr->name && cur->fabric_attr &&
            cur->fabric_attr->name) {

            std::string device_name = cur->domain_attr->name;
            std::string provider_name = cur->fabric_attr->prov_name;

            NIXL_TRACE << "Found device - domain: " << device_name << ", provider=" << provider_name
                       << ", ep_type=" << cur->ep_attr->type << ", caps=" << std::hex << cur->caps
                       << std::dec;

            if (provider_device_map.find(provider_name) == provider_device_map.end()) {
                provider_device_map[provider_name] = {};
            }
            provider_device_map[provider_name].push_back(device_name);
        }
    }

    fi_freeinfo(info);
    fi_freeinfo(hints);

    for (auto device_list : provider_device_map) {
        for (auto device : device_list.second) {
            NIXL_TRACE << "provider=" << device_list.first << ", device=" << device;
        }
    }

    // Provider selection priority:
    // 1. EFA (AWS Elastic Fabric Adapter)
    // 2. verbs;ofi_rxm (explicit verbs with RXM)
    // 3. verbs (plain verbs)
    // 4. sockets (TCP fallback)

    if (provider_device_map.find("efa") != provider_device_map.end()) {
        return {"efa", provider_device_map["efa"]};
    } else if (provider_device_map.find("verbs;ofi_rxm") != provider_device_map.end()) {
        // Explicit verbs with RXM
        NIXL_INFO << "Using verbs with RXM for RDM endpoint support";
        return {"verbs;ofi_rxm", provider_device_map["verbs;ofi_rxm"]};
    } else if (provider_device_map.find("verbs") != provider_device_map.end()) {
        // Plain verbs - might not support RDM, but try it
        NIXL_WARN << "Using plain verbs provider - may not support RDM endpoints. "
                  << "Consider setting FI_PROVIDER=verbs;ofi_rxm for RDM support";
        return {"verbs", provider_device_map["verbs"]};
    } else if (provider_device_map.find("sockets") != provider_device_map.end()) {
        return {"sockets", {provider_device_map["sockets"][0]}};
    }

    NIXL_WARN << "No network devices found with any provider";
    return {"none", {}};
}

std::string
hexdump(const void *data) {
    static constexpr uint HEXDUMP_MAX_LENGTH = 56;
    std::stringstream ss;
    ss.str().reserve(HEXDUMP_MAX_LENGTH * 3);
    const unsigned char *bytes = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < HEXDUMP_MAX_LENGTH; ++i) {
        ss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(bytes[i]) << " ";
    }
    return ss.str();
}

// Thread-safe atomic counters for optimized ID generation
static std::atomic<uint16_t> g_xfer_id_counter{1}; // 16-bit XFER_ID counter, start from 1
static std::atomic<uint8_t> g_seq_id_counter{0}; // 4-bit SEQ_ID counter, start from 0

uint16_t
getNextXferId() {
    uint16_t xfer_id = g_xfer_id_counter.fetch_add(1);

    // Handle wraparound: 16-bit field can hold 0 to 65,535
    if (xfer_id > NIXL_XFER_ID_MASK) {
        // Reset counter atomically and get a fresh ID
        uint16_t expected = xfer_id;
        while (expected > NIXL_XFER_ID_MASK &&
               !g_xfer_id_counter.compare_exchange_weak(expected, 1)) {
            expected = g_xfer_id_counter.load();
        }
        xfer_id = g_xfer_id_counter.fetch_add(1);
        // Ensure we don't exceed the mask after reset
        if (xfer_id > NIXL_XFER_ID_MASK) {
            xfer_id = 1;
        }
    }

    return xfer_id;
}

uint8_t
getNextSeqId() {
    uint8_t seq_id = g_seq_id_counter.fetch_add(1);

    // Handle wraparound: 4-bit field can hold 0 to 15
    if (seq_id > NIXL_SEQ_ID_MASK) {
        // Reset counter atomically and get a fresh ID
        uint8_t expected = seq_id;
        while (expected > NIXL_SEQ_ID_MASK &&
               !g_seq_id_counter.compare_exchange_weak(expected, 0)) {
            expected = g_seq_id_counter.load();
        }
        seq_id = g_seq_id_counter.fetch_add(1);
        // Ensure we don't exceed the mask after reset
        if (seq_id > NIXL_SEQ_ID_MASK) {
            seq_id = 0;
        }
    }

    return seq_id;
}

void
resetSeqId() {
    // Reset SEQ_ID counter for new postXfer
    g_seq_id_counter.store(0);
}

} // namespace LibfabricUtils
