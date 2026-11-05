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
 * nfi_hmem vendor: AMD ROCr, through the HSA runtime (libhsa-runtime64), which a ROCm build links.
 * Compiles to nothing without NIXL_HAVE_ROCM.
 *
 * Mirrors libfabric's hmem_rocr.c, so both attribute a pointer and number a device alike:
 * - a pointer is ROCr memory when hsa_amd_pointer_info() knows it and its owner is a GPU agent,
 *   which covers HIP allocations and memory allocated through HSA directly;
 * - a device ordinal is the owner's index among the GPU agents in hsa_iterate_agents() order, the
 *   value libfabric expects in fi_mr_attr device.rocr (ROCR_VISIBLE_DEVICES filters it, but
 *   HIP_VISIBLE_DEVICES does not).
 *
 * No scope hooks: HSA calls name their agent explicitly, so nothing needs a current device.
 */

#include "nfi_hmem_vendor.h"

/****************************************
 * AMD ROCr (HSA)
 *****************************************/

#ifdef NIXL_HAVE_ROCM

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

/* GPU agents in hsa_iterate_agents() order, indexed by device ordinal; libfabric's cap. */
#define NFI_ROCR_MAX_AGENTS 64
static hsa_agent_t nfi_rocr_gpu_agents[NFI_ROCR_MAX_AGENTS];
static size_t nfi_rocr_num_gpu;

static hsa_status_t
nfi_rocr_agent_callback(hsa_agent_t agent, void *data) {
    (void)data;
    hsa_device_type_t type;

    if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) == HSA_STATUS_SUCCESS &&
        type == HSA_DEVICE_TYPE_GPU && nfi_rocr_num_gpu < NFI_ROCR_MAX_AGENTS) {
        nfi_rocr_gpu_agents[nfi_rocr_num_gpu++] = agent;
    }
    return HSA_STATUS_SUCCESS;
}

bool
nfi_rocr_init(void) {
    /* Reference-counted, so this does not disturb a runtime HIP has already initialised. */
    const hsa_status_t ret = hsa_init();
    if (ret != HSA_STATUS_SUCCESS) {
        /* HSA_STATUS_ERROR_OUT_OF_RESOURCES is how ROCr reports no supported device. */
        nfi_log(ret == HSA_STATUS_ERROR_OUT_OF_RESOURCES ? NFI_HMEM_LOG_DEBUG : NFI_HMEM_LOG_WARN,
                "hsa_init failed (%d), ROCr detection disabled",
                static_cast<int>(ret));
        return false;
    }

    nfi_rocr_num_gpu = 0;
    const hsa_status_t iter = hsa_iterate_agents(nfi_rocr_agent_callback, nullptr);
    if ((iter != HSA_STATUS_SUCCESS && iter != HSA_STATUS_INFO_BREAK) || nfi_rocr_num_gpu == 0) {
        nfi_log(NFI_HMEM_LOG_DEBUG, "no HSA GPU agents found, ROCr detection disabled");
        (void)hsa_shut_down();
        return false;
    }
    nfi_log(NFI_HMEM_LOG_INFO, "ROCr detection enabled with %zu GPU agent(s)", nfi_rocr_num_gpu);
    return true;
}

/* The ordinal of GPU agent @p agent, or false if it is not one of the GPU agents. */
static bool
nfi_rocr_agent_ordinal(hsa_agent_t agent, uint64_t *ordinal) {
    for (size_t i = 0; i < nfi_rocr_num_gpu; i++) {
        if (nfi_rocr_gpu_agents[i].handle == agent.handle) {
            *ordinal = i;
            return true;
        }
    }
    return false;
}

static void
nfi_rocr_fill_bus(struct nfi_hmem_bus_id *bus, enum nfi_hmem_bus_kind *kind, hsa_agent_t agent) {
    uint32_t domain = 0, bdfid = 0;

    if (hsa_agent_get_info(agent,
                           static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DOMAIN),
                           &domain) != HSA_STATUS_SUCCESS ||
        hsa_agent_get_info(agent,
                           static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_BDFID),
                           &bdfid) != HSA_STATUS_SUCCESS) {
        return;
    }

    /* BDFID is KFD's location ID: bus << 8 | device << 3 | function. */
    *kind = NFI_HMEM_BUS_ACCELERATOR;
    bus->domain = domain;
    bus->bus = static_cast<uint8_t>((bdfid >> 8) & 0xff);
    bus->slot = static_cast<uint8_t>((bdfid >> 3) & 0x1f);
    bus->func = static_cast<uint8_t>(bdfid & 0x7);
}

bool
nfi_rocr_query(const void *addr, struct nfi_hmem_info *info) {
    hsa_amd_pointer_info_t ptr_info = {};
    ptr_info.size = sizeof(ptr_info);

    if (hsa_amd_pointer_info(addr, &ptr_info, nullptr, nullptr, nullptr) != HSA_STATUS_SUCCESS ||
        ptr_info.type == HSA_EXT_POINTER_TYPE_UNKNOWN) {
        return false;
    }

    uint64_t ordinal = 0;
    if (!nfi_rocr_agent_ordinal(ptr_info.agentOwner, &ordinal)) {
        /* Not owned by a GPU agent, which libfabric also declines. */
        return false;
    }

    info->iface = FI_HMEM_ROCR;
    info->device = ordinal;
    nfi_rocr_fill_bus(&info->bus, &info->bus_kind, ptr_info.agentOwner);
    return true;
}

size_t
nfi_rocr_device_bdfs(struct nfi_hmem_bus_id *out, size_t max) {
    size_t found = 0;

    for (size_t i = 0; i < nfi_rocr_num_gpu; i++) {
        struct nfi_hmem_bus_id bus = {};
        enum nfi_hmem_bus_kind kind = NFI_HMEM_BUS_NONE;
        nfi_rocr_fill_bus(&bus, &kind, nfi_rocr_gpu_agents[i]);
        if (kind == NFI_HMEM_BUS_NONE) {
            continue;
        }
        if (out && found < max) {
            out[found] = bus;
        }
        found++;
    }
    return found;
}

#endif /* NIXL_HAVE_ROCM */
