/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Intel Corporation. All rights reserved.
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

#include "ofi_backend.h"
#include "common/nixl_log.h"
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_endpoint.h>

nixlOFI_Engine::nixlOFI_Engine(const nixlBackendInitParams* init_params) :
    nixlBackendEngine(init_params),
    fabric(nullptr),
    domain(nullptr),
    ep(nullptr),
    cq(nullptr),
    eq(nullptr),
    pep(nullptr),
    fi(nullptr),
    cached_provider_info(nullptr),
    av(nullptr),
    is_connectionless(false),
    eq_thread_stop(false),
    eq_timeout_ms(100),
    hmem_ze_supported(false),
    hmem_cuda_supported(false),
    hmem_synapseai_supported(false)
{
    local_agent_name = init_params->localAgent;
    struct fi_info *hints = nullptr;
    struct fi_info *info = nullptr;
    int ret = 0;

    // use FI_PROVIDER environment variable or fall back to sensible defaults
    const char* env_provider = getenv("FI_PROVIDER");
    if (env_provider) {
        provider_name = env_provider;
        NIXL_DEBUG << "Using FI_PROVIDER environment variable: " << provider_name;
    } else {
        // Default to verbs;ofi_rxm for good RDMA performance with RDM semantics
        provider_name = "verbs;ofi_rxm";
        NIXL_DEBUG << "Using default provider: " << provider_name;
    }

    // get EQ timeout parameter (0-60 seconds max)
    getLongParam(init_params, "eq_timeout_ms", eq_timeout_ms, 0, 60000);

    hints = fi_allocinfo();
    if (!hints) {
        initErr = true;
        NIXL_ERROR << "fi_allocinfo failed";
        return;
    }

    // FI_HMEM for heterogeneous memory support (GPU memory)
    // check if HMEM support is needed via parameter (default: false for DRAM-only usage)
    std::string enable_hmem = "false";
    getStringParam(init_params, "enable_hmem", enable_hmem);
    bool need_hmem = (enable_hmem == "true" || enable_hmem == "1");

    NIXL_INFO << "HMEM support requested: " << (need_hmem ? "YES" : "NO");

    if (need_hmem) {
        hints->caps |= FI_HMEM;
        NIXL_INFO << "Adding FI_HMEM to hints->caps for GPU memory support";
    } else {
        NIXL_INFO << "Skipping FI_HMEM test - DRAM memory only";
    }

    // configure hints based on provider - keep it simple with good defaults
    configureHintsForProvider(hints, provider_name);

    // debug print all hints
    NIXL_INFO << "=== constructor: fi_getinfo hints ===";
    NIXL_INFO << "provider name: " << hints->fabric_attr->prov_name;
    NIXL_INFO << "caps: " << fi_tostr(&hints->caps, FI_TYPE_CAPS);
    NIXL_INFO << "mode: " << fi_tostr(&hints->mode, FI_TYPE_MODE);
    NIXL_INFO << "ep_attr->type: " << fi_tostr(&hints->ep_attr->type, FI_TYPE_EP_TYPE);
    NIXL_INFO << "domain_attr->mr_mode: " << fi_tostr(&hints->domain_attr->mr_mode, FI_TYPE_MR_MODE);
    NIXL_INFO << "domain_attr->resource_mgmt: " << hints->domain_attr->resource_mgmt;
    NIXL_INFO << "addr_format: " << fi_tostr(&hints->addr_format, FI_TYPE_ADDR_FORMAT);
    NIXL_INFO << "========================";

    // let libfabric choose optimal settings; only override if explicitly needed
    // (removed provider-specific tuning - use environment variables instead)

    ret = fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0, hints, &info);
    if (ret) {
        NIXL_ERROR << "fi_getinfo failed: " << fi_strerror(-ret);
        goto cleanup_getinfo;
    }

    // use the first provider returned by fi_getinfo (highest performance)
    fi = info;
    if (!fi) {
        NIXL_ERROR << "No providers returned by fi_getinfo";
        goto cleanup_getinfo;
    }
    
    NIXL_DEBUG << "Selected provider: " << (fi->fabric_attr->prov_name ? fi->fabric_attr->prov_name : "unknown")
               << " with endpoint type: " << fi_tostr(&fi->ep_attr->type, FI_TYPE_EP_TYPE);

    // connectionless provider?
    is_connectionless = isConnectionlessProvider();

    // detect HMEM capabilities for this provider
    detectHmemCapabilities(fi, provider_name, hmem_cuda_supported,
                           hmem_ze_supported, hmem_synapseai_supported);

    ret = fi_fabric(fi->fabric_attr, &fabric, nullptr);
    if (ret) {
        NIXL_ERROR << "fi_fabric failed: " << fi_strerror(-ret);
        goto cleanup_getinfo;
    }

    ret = fi_domain(fabric, fi, &domain, nullptr);
    if (ret) {
        NIXL_ERROR << "fi_domain failed: " << fi_strerror(-ret);
        goto cleanup_fabric;
    }

    if (setupEndpoint(!is_connectionless) != NIXL_SUCCESS) {
        goto cleanup_domain;
    }

    // get local address
    if (getEndpointAddress(ep, local_addr) != NIXL_SUCCESS) {
        goto cleanup_ep;
    }

    // cache provider use in connect()
    cached_provider_info = fi_dupinfo(fi);
    if (!cached_provider_info) {
        NIXL_WARN << "Failed to duplicate provider info for caching";
    }

    fi_freeinfo(hints);
    fi_freeinfo(info);

    // start event loop thread for connection-oriented providers
    if (!is_connectionless) {
        eq_thread = std::thread(&nixlOFI_Engine::eq_event_loop, this);
    }
    return;

cleanup_ep:
    if (pep)    { fi_close(&pep->fid);    pep = nullptr; }
    if (ep)     { fi_close(&ep->fid);     ep = nullptr; }
    if (cq)     { fi_close(&cq->fid);     cq = nullptr; }
    if (eq)     { fi_close(&eq->fid);     eq = nullptr; }
    if (av)     { fi_close(&av->fid);     av = nullptr; }
cleanup_domain:
    if (domain) { fi_close(&domain->fid); domain = nullptr; }
cleanup_fabric:
    if (fabric) { fi_close(&fabric->fid); fabric = nullptr; }
cleanup_getinfo:
    if (hints)  { fi_freeinfo(hints);     hints = nullptr; }
    if (info)   { fi_freeinfo(info);      info = nullptr; }
    initErr = true;
}

// Parameter extraction helper methods
void nixlOFI_Engine::getStringParam(const nixlBackendInitParams* init_params, const std::string& key, std::string& value) {
    auto it = init_params->customParams->find(key);
    if (it != init_params->customParams->end()) {
        value = it->second;
    }
}

void nixlOFI_Engine::getLongParam(const nixlBackendInitParams* init_params, const std::string& key, long& value, long min_val, long max_val) {
    auto it = init_params->customParams->find(key);
    if (it != init_params->customParams->end()) {
        try {
            long parsed_val = std::stol(it->second);
            if (parsed_val >= min_val && parsed_val <= max_val) {
                value = parsed_val;
            } else {
                NIXL_WARN << key << " out of range [" << min_val << "-" << max_val << "]: " << parsed_val << ", using default " << value;
            }
        } catch (const std::exception& e) {
            NIXL_WARN << "Invalid " << key << " parameter: " << it->second << ", using default " << value;
        }
    }
}

void nixlOFI_Engine::getSizeTParam(const nixlBackendInitParams* init_params, const std::string& key, size_t& value) {
    auto it = init_params->customParams->find(key);
    if (it != init_params->customParams->end()) {
        try {
            size_t parsed_val = std::stoull(it->second);
            value = parsed_val;
            NIXL_DEBUG << "Set " << key << " to " << value;
        } catch (const std::exception& e) {
            NIXL_WARN << "Invalid " << key << ": " << it->second << ", keeping default " << value;
        }
    }
}

// predefined provider configurations based on fabtests test_configs
const nixlOFI_Engine::ProviderConfig nixlOFI_Engine::SUPPORTED_PROVIDERS[] = {
    {
        "shm",
        FI_EP_RDM,
        FI_MSG | FI_RMA | FI_READ | FI_WRITE,
        FI_CONTEXT | FI_CONTEXT2,
        FI_MR_VIRT_ADDR,
        FI_RM_UNSPEC,
        {0, 0, 0, 0, 0, 0, 0, 0, FI_TC_UNSPEC}, // tx_attr defaults
        {0, 0, 0, 0, 0, 0}, // rx_attr defaults
        FI_FORMAT_UNSPEC,
        FI_PROGRESS_AUTO,
        FI_PROGRESS_AUTO
    },
    {
        "tcp",
        FI_EP_MSG,
        FI_MSG | FI_RMA | FI_READ | FI_WRITE,
        FI_CONTEXT | FI_CONTEXT2,
        0, // let provider choose mr_mode
        FI_RM_ENABLED,
        {0, 0, 0, 0, 0, 0, 0, 0, FI_TC_BULK_DATA}, // tx_attr with bulk data class
        {0, 0, 0, 0, 0, 0}, // rx_attr defaults
        FI_FORMAT_UNSPEC,
        FI_PROGRESS_MANUAL,
        FI_PROGRESS_MANUAL
    },
    {
        "verbs",
        FI_EP_MSG,
        FI_MSG | FI_RMA | FI_READ | FI_WRITE,
        FI_CONTEXT | FI_CONTEXT2,
        FI_MR_LOCAL | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY,
        FI_RM_ENABLED,
        {0, 0, 0, 0, 0, 0, 0, 0, FI_TC_BULK_DATA}, // tx_attr with bulk data class
        {0, 0, 0, 0, 0, 0}, // rx_attr defaults
        FI_SOCKADDR_IB,
        FI_PROGRESS_MANUAL,
        FI_PROGRESS_MANUAL
    },
    {
        "verbs;ofi_rxm",
        FI_EP_RDM,
        FI_MSG, // match fabtests exactly - only FI_MSG capability
        FI_CONTEXT | FI_CONTEXT2,
        FI_MR_LOCAL | FI_MR_RAW | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_ENDPOINT, // match fabtests mr_mode 636
        FI_RM_ENABLED,
        {0, 0, 0, 0, 0, 0, 0, 0, FI_TC_BULK_DATA}, // tx_attr with bulk data class like fabtests
        {0, 0, 0, 0, 0, 0}, // rx_attr defaults
        FI_FORMAT_UNSPEC,
        FI_PROGRESS_AUTO,
        FI_PROGRESS_AUTO
    }
};

const size_t nixlOFI_Engine::NUM_SUPPORTED_PROVIDERS =
    sizeof(SUPPORTED_PROVIDERS) / sizeof(SUPPORTED_PROVIDERS[0]);

const nixlOFI_Engine::ProviderConfig* nixlOFI_Engine::findProviderConfig(const std::string& provider_name) {
    for (size_t i = 0; i < NUM_SUPPORTED_PROVIDERS; ++i) {
        if (SUPPORTED_PROVIDERS[i].name == provider_name) {
            return &SUPPORTED_PROVIDERS[i];
        }
    }
    return nullptr;
}

void nixlOFI_Engine::configureHintsForProvider(struct fi_info* hints, const std::string& provider_name) {
    const auto* config = findProviderConfig(provider_name);

    if (!config) {
        // if the provider is not in our list, use the verbs;ofi_rxm config as a safe default
        config = findProviderConfig("verbs;ofi_rxm");
        NIXL_DEBUG << "Unknown provider '" << provider_name << "', using verbs;ofi_rxm config as a fallback.";
    } else {
        NIXL_DEBUG << "Using predefined config for provider: " << provider_name;
    }

    // apply the configuration from the data structure
    hints->ep_attr->type = config->ep_type;
    hints->domain_attr->resource_mgmt = config->resource_mgmt;
    hints->caps = config->caps;
    hints->mode = config->mode;

    if (config->mr_mode != 0) {
        hints->domain_attr->mr_mode = config->mr_mode;
    }

    // apply tx/rx attributes
    if (config->tx_attr.tclass != 0) {
        hints->tx_attr->tclass = config->tx_attr.tclass;
    }
    // other tx_attr fields can be added here as needed
    
    // rx_attr fields can be added here as needed

    // address format - only set if not UNSPEC
    if (config->addr_format != FI_FORMAT_UNSPEC) {
        hints->addr_format = config->addr_format;
    }

    // progress models - only set if not UNSPEC 
    if (config->data_progress != FI_PROGRESS_UNSPEC) {
        hints->domain_attr->data_progress = config->data_progress;
    }
    if (config->control_progress != FI_PROGRESS_UNSPEC) {
        hints->domain_attr->control_progress = config->control_progress;
    }

    // enable shared RX context for verbs providers to use XRC endpoints like fabtests
    if (provider_name.find("verbs") != std::string::npos) {
        hints->ep_attr->rx_ctx_cnt = FI_SHARED_CONTEXT;
    }

    // always set the provider name in the hints
    if (hints->fabric_attr->prov_name) free(hints->fabric_attr->prov_name);
    hints->fabric_attr->prov_name = strdup(provider_name.c_str());
}

nixlOFI_Engine::~nixlOFI_Engine() {
    if (!is_connectionless) {
        eq_thread_stop = true;
        if (eq_thread.joinable()) {
            // wake up the EQ thread to ensure it exits
            if (eq) {
                uint32_t event;
                fi_eq_read(eq, &event, nullptr, 0, 0);
            }
            eq_thread.join();
        }
    }

    // close connected endpoints
    for (auto const& [key, val] : connected_eps) {
        fi_close(&val->fid);
    }

    if (pep)    { fi_close(&pep->fid);    pep = nullptr; }
    if (ep)     { fi_close(&ep->fid);     ep = nullptr; }
    if (cq)     { fi_close(&cq->fid);     cq = nullptr; }
    if (eq)     { fi_close(&eq->fid);     eq = nullptr; }
    if (av)     { fi_close(&av->fid);     av = nullptr; }
    if (domain) { fi_close(&domain->fid); domain = nullptr; }
    if (fabric) { fi_close(&fabric->fid); fabric = nullptr; }
    if (cached_provider_info) fi_freeinfo(cached_provider_info);
}

bool nixlOFI_Engine::supportsNotif() const {
    return false;
}

bool nixlOFI_Engine::supportsRemote() const {
    return true;
}

bool nixlOFI_Engine::supportsLocal() const {
    return false;
}

bool nixlOFI_Engine::supportsProgTh() const {
    return true;
}

nixl_mem_list_t nixlOFI_Engine::getSupportedMems() const {
    nixl_mem_list_t mems;
    mems.push_back(DRAM_SEG);
    if (hmem_cuda_supported) {
        mems.push_back(VRAM_SEG);
    }
    if (hmem_synapseai_supported) {
        mems.push_back(VRAM_SEG);
    }
    if (hmem_ze_supported) {
        mems.push_back(VRAM_SEG);
    }
    return mems;
}

nixl_status_t nixlOFI_Engine::connect(const std::string &remote_agent) {
    std::lock_guard<std::mutex> lock(ep_lock);

    if (is_connectionless) {
        // for connectionless providers like shm: insert remote address into av
        if (shm_addrs.count(remote_agent)) {
            NIXL_DEBUG << "Already have address mapping for " << remote_agent;
            return NIXL_SUCCESS;
        }

        auto remote_addr_it = remote_addrs.find(remote_agent);
        if (remote_addr_it == remote_addrs.end()) {
            NIXL_ERROR << "Remote address for " << remote_agent << " not found.";
            return NIXL_ERR_NOT_FOUND;
        }

        fi_addr_t addr;
        int ret = fi_av_insert(av, remote_addr_it->second.data(), 1, &addr, 0, nullptr);
        if (ret != 1) {
            NIXL_ERROR << "fi_av_insert failed: " << fi_strerror(-ret);
            return NIXL_ERR_BACKEND;
        }

        shm_addrs[remote_agent] = addr;
        NIXL_DEBUG << "OFI backend: Added address mapping for " << remote_agent;
        return NIXL_SUCCESS;
    }

    // connection-oriented logic
    if (connected_eps.count(remote_agent)) {
        NIXL_DEBUG << "Already connected to " << remote_agent;
        return NIXL_SUCCESS;
    }

    auto remote_addr_it = remote_addrs.find(remote_agent);
    if (remote_addr_it == remote_addrs.end()) {
        NIXL_ERROR << "Remote address for " << remote_agent << " not found.";
        return NIXL_ERR_NOT_FOUND;
    }
    const std::string &remote_addr_str = remote_addr_it->second;

    // create copy of provider info to avoid shared state issues
    struct fi_info *remote_fi = fi_dupinfo(cached_provider_info);
    if (!remote_fi) {
        NIXL_ERROR << "Failed to duplicate provider info for remote agent";
        return NIXL_ERR_BACKEND;
    }

    // update dest_addr for this connection
    remote_fi->dest_addr = (void*)remote_addr_str.c_str();
    // let libfabric use the auto-negotiated address format from fi_getinfo

    fid_ep *remote_ep = nullptr;
    int ret = fi_endpoint(domain, remote_fi, &remote_ep, nullptr);
    if (ret) {
        NIXL_ERROR << "fi_endpoint for remote failed: " << fi_strerror(-ret);
        fi_freeinfo(remote_fi);
        return NIXL_ERR_BACKEND;
    }

    ret = fi_ep_bind(remote_ep, &cq->fid, FI_SEND | FI_RECV);
    if (ret) {
        NIXL_ERROR << "fi_ep_bind to CQ for remote failed: " << fi_strerror(-ret);
        fi_close(&remote_ep->fid);
        fi_freeinfo(remote_fi);
        return NIXL_ERR_BACKEND;
    }

    ret = fi_ep_bind(remote_ep, &eq->fid, FI_SOURCE | FI_RMA | FI_MSG);
    if (ret) {
        NIXL_ERROR << "fi_ep_bind to EQ for remote failed: " << fi_strerror(-ret);
        fi_close(&remote_ep->fid);
        fi_freeinfo(remote_fi);
        return NIXL_ERR_BACKEND;
    }

    ret = fi_enable(remote_ep);
    if (ret) {
        NIXL_ERROR << "fi_enable for remote failed: " << fi_strerror(-ret);
        fi_close(&remote_ep->fid);
        fi_freeinfo(remote_fi);
        return NIXL_ERR_BACKEND;
    }

    ret = fi_connect(remote_ep, remote_fi->dest_addr, local_agent_name.c_str(), local_agent_name.length() + 1);
    if (ret) {
        NIXL_ERROR << "fi_connect failed: " << fi_strerror(-ret);
        fi_close(&remote_ep->fid);
        fi_freeinfo(remote_fi);
        return NIXL_ERR_BACKEND;
    }

    // wait for connection to complete via EQ
    struct fi_eq_cm_entry entry;
    uint32_t event;
    ssize_t n_events = fi_eq_read(eq, &event, &entry, 1, -1);
    if (n_events < 0) {
        NIXL_ERROR << "fi_eq_read failed during connect: " << fi_strerror(-n_events);
        fi_close(&remote_ep->fid);
        fi_freeinfo(remote_fi);
        return NIXL_ERR_BACKEND;
    }
    if (event != FI_CONNECTED || entry.fid != &remote_ep->fid) {
        NIXL_ERROR << "Unexpected EQ event during connect: " << event;
        fi_close(&remote_ep->fid);
        fi_freeinfo(remote_fi);
        return NIXL_ERR_BACKEND;
    }

    connected_eps[remote_agent] = remote_ep;
    fi_freeinfo(remote_fi);

    NIXL_DEBUG << "OFI backend: Connected to " << remote_agent;
    return NIXL_SUCCESS;
}

nixl_status_t nixlOFI_Engine::disconnect(const std::string &remote_agent) {
    std::lock_guard<std::mutex> lock(ep_lock);

    if (is_connectionless) {
        // connectionless provider, remove address mapping
        auto it = shm_addrs.find(remote_agent);
        if (it == shm_addrs.end()) {
            NIXL_WARN << "OFI backend: No address mapping for " << remote_agent;
            return NIXL_ERR_NOT_FOUND;
        }

        int ret = fi_av_remove(av, &it->second, 1, 0);
        if (ret) {
            NIXL_ERROR << "fi_av_remove failed: " << fi_strerror(-ret);
            return NIXL_ERR_BACKEND;
        }

        shm_addrs.erase(it);
        NIXL_DEBUG << "OFI backend: Removed address mapping for " << remote_agent;
        return NIXL_SUCCESS;
    }

    // connection-oriented case
    auto it = connected_eps.find(remote_agent);
    if (it == connected_eps.end()) {
        NIXL_WARN << "OFI backend: No active connection to " << remote_agent;
        return NIXL_ERR_NOT_FOUND;
    }

    // Store endpoint before erasing to ensure proper cleanup even if fi_close fails
    fid_ep* ep_to_close = it->second;
    connected_eps.erase(it);
    
    int ret = fi_close(&ep_to_close->fid);
    if (ret) {
        NIXL_ERROR << "fi_close (remote_ep) failed: " << fi_strerror(-ret);
        return NIXL_ERR_BACKEND;
    }
    NIXL_DEBUG << "OFI backend: Disconnected from " << remote_agent;
    return NIXL_SUCCESS;
}

nixl_status_t nixlOFI_Engine::registerMem(const nixlBlobDesc &mem,
                                     const nixl_mem_t &nixl_mem,
                                     nixlBackendMD* &out) {
    nixlOFI_Metadata *ofi_meta = new nixlOFI_Metadata();
    if (!ofi_meta) {
        return NIXL_ERR_BACKEND;
    }

    int ret = 0;
    if (nixl_mem == DRAM_SEG) {
        ret = fi_mr_reg(domain, reinterpret_cast<void*>(mem.addr), mem.len,
                       FI_REMOTE_READ | FI_REMOTE_WRITE | FI_SEND | FI_RECV,
                       0, 0, 0, &ofi_meta->mr, nullptr);
    } else if (nixl_mem == VRAM_SEG) {
        struct fi_mr_attr mr_attr = {};
        struct iovec iov = {};

        iov.iov_base = reinterpret_cast<void*>(mem.addr);
        iov.iov_len = mem.len;

        mr_attr.mr_iov = &iov;
        mr_attr.iov_count = 1;
        mr_attr.access = FI_REMOTE_READ | FI_REMOTE_WRITE | FI_SEND | FI_RECV;

        // prioritize device interfaces based on availability and device ID
        if (hmem_ze_supported && mem.devId >= 0) {
            mr_attr.iface = FI_HMEM_ZE;
            mr_attr.device.ze = mem.devId;
            NIXL_DEBUG << "Using ZE HMEM interface for device " << mem.devId;
        } else if (hmem_synapseai_supported && mem.devId >= 0) {
            mr_attr.iface = FI_HMEM_SYNAPSEAI;
            mr_attr.device.synapseai = mem.devId;
            NIXL_DEBUG << "Using SynapseAI HMEM interface for device " << mem.devId;
        } else if (hmem_cuda_supported && mem.devId >= 0) {
            mr_attr.iface = FI_HMEM_CUDA;
            mr_attr.device.cuda = mem.devId;
            NIXL_DEBUG << "Using CUDA HMEM interface for device " << mem.devId;
        } else {
            NIXL_ERROR << "VRAM memory requested but no supported HMEM interface available. "
                      << "CUDA: " << hmem_cuda_supported 
                      << ", ZE: " << hmem_ze_supported 
                      << ", SynapseAI: " << hmem_synapseai_supported
                      << ", DeviceID: " << mem.devId;
            delete ofi_meta;
            return NIXL_ERR_NOT_SUPPORTED;
        }

        ret = fi_mr_regattr(domain, &mr_attr, 0, &ofi_meta->mr);
    } else {
        NIXL_ERROR << "Unsupported memory type: " << nixl_mem;
        delete ofi_meta;
        return NIXL_ERR_NOT_SUPPORTED;
    }

    if (ret) {
        NIXL_ERROR << "fi_mr_reg failed: " << fi_strerror(-ret);
        delete ofi_meta;
        return NIXL_ERR_BACKEND;
    }

    ofi_meta->desc = fi_mr_desc(ofi_meta->mr);
    if (!ofi_meta->desc) {
        NIXL_ERROR << "fi_mr_desc failed";
        fi_close(&ofi_meta->mr->fid);
        delete ofi_meta;
        return NIXL_ERR_BACKEND;
    }

    out = ofi_meta;
    return NIXL_SUCCESS;
}

nixl_status_t nixlOFI_Engine::deregisterMem(nixlBackendMD *meta) {
    nixlOFI_Metadata *ofi_meta = static_cast<nixlOFI_Metadata*>(meta);
    if (!ofi_meta) {
        return NIXL_ERR_INVALID_PARAM;
    }

    // Only close mr for local metadata - remote metadata has mr = nullptr
    if (ofi_meta->mr) {
        int ret = fi_close(&ofi_meta->mr->fid);
        if (ret) {
            NIXL_ERROR << "fi_close (mr) failed: " << fi_strerror(-ret);
            return NIXL_ERR_BACKEND;
        }
    }

    delete ofi_meta;
    return NIXL_SUCCESS;
}

nixl_status_t nixlOFI_Engine::unloadMD(nixlBackendMD* input) {
    return deregisterMem(input);
}

nixl_status_t nixlOFI_Engine::prepXfer(const nixl_xfer_op_t &operation,
                                  const nixl_meta_dlist_t &local,
                                  const nixl_meta_dlist_t &remote,
                                  const std::string &remote_agent,
                                  nixlBackendReqH* &handle,
                                  const nixl_opt_b_args_t* opt_args) const {
    return postXfer(operation, local, remote, remote_agent, handle, opt_args);
}

nixl_status_t nixlOFI_Engine::postXfer(const nixl_xfer_op_t &operation,
                                  const nixl_meta_dlist_t &local,
                                  const nixl_meta_dlist_t &remote,
                                  const std::string &remote_agent,
                                  nixlBackendReqH* &handle,
                                  const nixl_opt_b_args_t* opt_args) const {
    fid_ep *target_ep = ep;
    fi_addr_t dest_addr = FI_ADDR_UNSPEC;

    if (is_connectionless) {
        auto shm_it = shm_addrs.find(remote_agent);
        if (shm_it == shm_addrs.end()) {
            NIXL_ERROR << "OFI backend: No address mapping for " << remote_agent;
            return NIXL_ERR_NOT_FOUND;
        }
        dest_addr = shm_it->second;
    } else {
        auto it = connected_eps.find(remote_agent);
        if (it == connected_eps.end()) {
            NIXL_ERROR << "OFI backend: Not connected to " << remote_agent;
            return NIXL_ERR_NOT_FOUND;
        }
        target_ep = it->second;
    }

    if (local.descCount() != remote.descCount()) {
        NIXL_ERROR << "Mismatched descriptor counts: local=" << local.descCount()
                   << ", remote=" << remote.descCount();
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlOFI_Request *ofi_req = new nixlOFI_Request();
    if (!ofi_req) {
        return NIXL_ERR_BACKEND;
    }
    ofi_req->cq = cq;

    int ret = 0;
    for (size_t i = 0; i < static_cast<size_t>(local.descCount()); ++i) {
        const nixlMetaDesc &local_desc = local[i];
        const nixlMetaDesc &remote_desc = remote[i];

        nixlOFI_Metadata *local_meta = static_cast<nixlOFI_Metadata*>(local_desc.metadataP);
        nixlOFI_Metadata *remote_meta = static_cast<nixlOFI_Metadata*>(remote_desc.metadataP);
        
        if (!local_meta || !remote_meta || !local_meta->mr) {
            NIXL_ERROR << "Invalid metadata or memory registration";
            delete ofi_req;
            return NIXL_ERR_INVALID_PARAM;
        }

        // Get remote memory key - either from mr or stored in desc field for remote metadata
        uint64_t remote_key;
        if (remote_meta->mr) {
            remote_key = fi_mr_key(remote_meta->mr);
        } else {
            // For remote metadata, key is stored in desc field
            remote_key = reinterpret_cast<uint64_t>(remote_meta->desc);
        }
        
        struct fi_rma_iov rma_iov = {
            .addr = (uint64_t)remote_desc.addr,
            .len = remote_desc.len,
            .key = remote_key
        };

        switch (operation) {
            case NIXL_READ:
                ret = fi_read(target_ep, reinterpret_cast<void*>(local_desc.addr),
                             local_desc.len, local_meta->desc, dest_addr,
                             rma_iov.addr, rma_iov.key, &ofi_req->wr_id);
                break;
            case NIXL_WRITE:
                ret = fi_write(target_ep, reinterpret_cast<void*>(local_desc.addr),
                              local_desc.len, local_meta->desc, dest_addr,
                              rma_iov.addr, rma_iov.key, &ofi_req->wr_id);
                break;
            default:
                NIXL_ERROR << "Unsupported operation type";
                delete ofi_req;
                return NIXL_ERR_NOT_SUPPORTED;
        }

        if (ret) {
            NIXL_ERROR << "OFI transfer failed: " << fi_strerror(-ret);
            delete ofi_req;
            return NIXL_ERR_BACKEND;
        }
    }

    handle = ofi_req;
    return NIXL_SUCCESS;

}


nixl_status_t nixlOFI_Engine::checkXfer(nixlBackendReqH* handle) const {
    nixlOFI_Request *ofi_req = static_cast<nixlOFI_Request*>(handle);
    if (!ofi_req) {
        return NIXL_ERR_INVALID_PARAM;
    }

    struct fi_cq_data_entry entry;
    int ret = fi_cq_read(ofi_req->cq, &entry, 1);
    if (ret == 1) {
        return NIXL_SUCCESS;
    } else if (ret == -FI_EAGAIN) {
        return NIXL_IN_PROG;
    } else if (ret < 0) {
        struct fi_cq_err_entry err_entry;
        int err_ret = fi_cq_readerr(ofi_req->cq, &err_entry, 0);
        if (err_ret > 0) {
            NIXL_ERROR << "CQ error: " << fi_strerror(err_entry.err) << " (" << err_entry.err << ")";
        } else {
            NIXL_ERROR << "fi_cq_read failed: " << fi_strerror(-ret);
        }
        return NIXL_ERR_BACKEND;
    }
    return NIXL_IN_PROG;
}

nixl_status_t nixlOFI_Engine::releaseReqH(nixlBackendReqH* handle) const {
    nixlOFI_Request *ofi_req = static_cast<nixlOFI_Request*>(handle);
    if (!ofi_req) {
        return NIXL_ERR_INVALID_PARAM;
    }
    delete ofi_req;
    return NIXL_SUCCESS;
}

nixl_status_t nixlOFI_Engine::getConnInfo(std::string &conn_info) const {
    conn_info = local_addr;
    return NIXL_SUCCESS;
}

nixl_status_t nixlOFI_Engine::loadRemoteConnInfo(const std::string &remote_agent, const std::string &conn_info) {
    std::lock_guard<std::mutex> lock(ep_lock);
    remote_addrs[remote_agent] = conn_info;
    return NIXL_SUCCESS;
}

nixl_status_t nixlOFI_Engine::getPublicData(const nixlBackendMD* meta, std::string &str) const {
    const nixlOFI_Metadata* ofi_meta = static_cast<const nixlOFI_Metadata*>(meta);
    if (!ofi_meta || !ofi_meta->mr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    
    // Serialize the memory registration key for remote access
    uint64_t mr_key = fi_mr_key(ofi_meta->mr);
    str = std::to_string(mr_key);
    return NIXL_SUCCESS;
}

nixl_status_t nixlOFI_Engine::loadRemoteMD(const nixlBlobDesc &input, const nixl_mem_t &nixl_mem,
                                           const std::string &remote_agent, nixlBackendMD* &output) {
    // Create a remote metadata object from the serialized public data
    nixlOFI_Metadata* remote_meta = new nixlOFI_Metadata();
    if (!remote_meta) {
        return NIXL_ERR_BACKEND;
    }
    
    // Parse the memory key from the metadata string
    try {
        // The metaInfo should contain the memory key as a string
        std::string key_str(input.metaInfo.data(), input.metaInfo.size());
        uint64_t remote_key = std::stoull(key_str);
        
        // For remote metadata, we don't have an actual mr object, just the key
        // Store the key for later use in RMA operations
        remote_meta->mr = nullptr;  // No local mr for remote metadata
        remote_meta->desc = reinterpret_cast<void*>(remote_key);  // Store key in desc field
        
        output = remote_meta;
        return NIXL_SUCCESS;
    } catch (const std::exception& e) {
        delete remote_meta;
        NIXL_ERROR << "Failed to parse remote memory key: " << e.what();
        return NIXL_ERR_INVALID_PARAM;
    }
}

void nixlOFI_Engine::eq_event_loop() {
    while (!eq_thread_stop) {
        struct fi_eq_cm_entry entry;
        uint32_t event;
        ssize_t ret = fi_eq_read(eq, &event, &entry, 1, eq_timeout_ms);

        if (ret == -FI_EAGAIN) {
            continue;
        } else if (ret < 0) {
            if (ret == -FI_EINTR && eq_thread_stop) {
                // interrupt
                break;
            }
            NIXL_ERROR << "fi_eq_read failed in event loop: " << fi_strerror(-ret);
            // TODO: error handling
            continue;
        }

        switch (event) {
            case FI_CONNREQ:
            {
                NIXL_DEBUG << "FI_CONNREQ event received";
                fid_ep *new_ep = nullptr;

                // accept
                int connreq_ret = fi_endpoint(domain, fi, &new_ep, nullptr);
                if (connreq_ret) {
                    NIXL_ERROR << "fi_endpoint for accepted connection failed: " << fi_strerror(-connreq_ret);
                    break;
                }
                connreq_ret = fi_ep_bind(new_ep, &cq->fid, FI_SEND | FI_RECV);
                if (connreq_ret) {
                    NIXL_ERROR << "fi_ep_bind to CQ for accepted connection failed: " << fi_strerror(-connreq_ret);
                    fi_close(&new_ep->fid);
                    break;
                }
                connreq_ret = fi_ep_bind(new_ep, &eq->fid, FI_SOURCE | FI_RMA | FI_MSG);
                if (connreq_ret) {
                    NIXL_ERROR << "fi_ep_bind to EQ for accepted connection failed: " << fi_strerror(-connreq_ret);
                    fi_close(&new_ep->fid);
                    break;
                }
                connreq_ret = fi_accept(new_ep, nullptr, 0);
                if (connreq_ret) {
                    NIXL_ERROR << "fi_accept failed: " << fi_strerror(-connreq_ret);
                    fi_close(&new_ep->fid);
                    break;
                }
                connreq_ret = fi_enable(new_ep);
                if (connreq_ret) {
                    NIXL_ERROR << "fi_enable for accepted connection failed: " << fi_strerror(-connreq_ret);
                    fi_close(&new_ep->fid);
                    break;
                }

                std::string remote_agent_name = "connected_agent_" + std::to_string(reinterpret_cast<uintptr_t>(new_ep));

                std::lock_guard<std::mutex> lock(ep_lock);
                connected_eps[remote_agent_name] = new_ep;
                NIXL_DEBUG << "Accepted connection from " << remote_agent_name;
                break;
            }
            case FI_CONNECTED:
                NIXL_DEBUG << "FI_CONNECTED event received for outgoing connection";
                // TODO: async model
                break;
            case FI_SHUTDOWN:
                NIXL_DEBUG << "FI_SHUTDOWN event received";
                {
                    std::lock_guard<std::mutex> lock(ep_lock);
                    for (auto it = connected_eps.begin(); it != connected_eps.end(); ++it) {
                        if (&it->second->fid == entry.fid) {
                            fi_close(&it->second->fid);
                            connected_eps.erase(it);
                            break;
                        }
                    }
                }
                break;
            default:
                NIXL_WARN << "Unhandled EQ event: " << event;
                break;
        }
    }
}

bool nixlOFI_Engine::isConnectionlessProvider() const {
    // ofi_rxm is always connectionless (RDM model) regardless of underlying provider
    if (provider_name.find("ofi_rxm") != std::string::npos) {
        return true;
    }
    
    if (provider_name == "shm" || provider_name == "udp") {
        return true;
    }
    
    // also check the actual provider name from libfabric
    if (fi && fi->fabric_attr && fi->fabric_attr->prov_name) {
        std::string actual_provider = fi->fabric_attr->prov_name;
        if (actual_provider.find("ofi_rxm") != std::string::npos ||
            actual_provider == "shm" || actual_provider == "udp" ||
            actual_provider.find("shm") != std::string::npos) {
            return true;
        }
    }
    
    return false;
}

nixl_status_t nixlOFI_Engine::setupEndpoint(bool connection_oriented) {
    int ret = 0;

    // create endpoint
    ret = fi_endpoint(domain, fi, &ep, nullptr);
    if (ret) {
        NIXL_ERROR << "fi_endpoint failed: " << fi_strerror(-ret);
        return NIXL_ERR_BACKEND;
    }

    // create and bind completion queue
    struct fi_cq_attr cq_attr = {};
    cq_attr.size = 128; // use fi->tx_attr->size + fi->rx_attr->size?
    cq_attr.format = FI_CQ_FORMAT_CONTEXT;
    ret = fi_cq_open(domain, &cq_attr, &cq, nullptr);
    if (ret) {
        NIXL_ERROR << "fi_cq_open failed: " << fi_strerror(-ret);
        goto cleanup_endpoint;
    }

    ret = fi_ep_bind(ep, &cq->fid, FI_SEND | FI_RECV);
    if (ret) {
        NIXL_ERROR << "fi_ep_bind to CQ failed: " << fi_strerror(-ret);
        goto cleanup_cq;
    }

    if (connection_oriented) {
        // event queue for connection management
        struct fi_eq_attr eq_attr = {};
        eq_attr.size = 64;
        eq_attr.wait_obj = FI_WAIT_UNSPEC;
        ret = fi_eq_open(fabric, &eq_attr, &eq, nullptr);
        if (ret) {
            NIXL_ERROR << "fi_eq_open failed: " << fi_strerror(-ret);
            goto cleanup_cq;
        }

        // bind endpoint to EQ for connection management
        ret = fi_ep_bind(ep, &eq->fid, 0);
        if (ret) {
            NIXL_ERROR << "fi_ep_bind to EQ failed: " << fi_strerror(-ret);
            goto cleanup_eq;
        }

        // create passive endpoint for listening
        ret = fi_passive_ep(fabric, fi, &pep, nullptr);
        if (ret) {
            NIXL_ERROR << "fi_passive_ep failed: " << fi_strerror(-ret);
            goto cleanup_eq;
        }

        ret = fi_pep_bind(pep, &eq->fid, 0);
        if (ret) {
            NIXL_ERROR << "fi_pep_bind to EQ failed: " << fi_strerror(-ret);
            goto cleanup_pep;
        }

        ret = fi_listen(pep);
        if (ret) {
            NIXL_ERROR << "fi_listen failed: " << fi_strerror(-ret);
            goto cleanup_pep;
        }
    } else {
        // address vector for connectionless communication
        struct fi_av_attr av_attr = {};
        av_attr.type = FI_AV_MAP;
        ret = fi_av_open(domain, &av_attr, &av, nullptr);
        if (ret) {
            NIXL_ERROR << "fi_av_open failed: " << fi_strerror(-ret);
            goto cleanup_cq;
        }

        ret = fi_ep_bind(ep, &av->fid, 0);
        if (ret) {
            NIXL_ERROR << "fi_ep_bind to AV failed: " << fi_strerror(-ret);
            goto cleanup_av;
        }

        ret = fi_enable(ep);
        if (ret) {
            NIXL_ERROR << "fi_enable failed: " << fi_strerror(-ret);
            goto cleanup_av;
        }
    }
    return NIXL_SUCCESS;

    // Cleanup in reverse order of allocation
cleanup_pep:
    if (pep) { fi_close(&pep->fid); pep = nullptr; }
cleanup_eq:
    if (eq) { fi_close(&eq->fid); eq = nullptr; }
cleanup_av:
    if (av) { fi_close(&av->fid); av = nullptr; }
cleanup_cq:
    if (cq) { fi_close(&cq->fid); cq = nullptr; }
cleanup_endpoint:
    if (ep) { fi_close(&ep->fid); ep = nullptr; }
    return NIXL_ERR_BACKEND;
}

nixl_status_t nixlOFI_Engine::getEndpointAddress(fid_ep* endpoint, std::string& address) {
    if (!endpoint) {
        return NIXL_ERR_INVALID_PARAM;
    }

    size_t addrlen = 256;
    std::vector<char> addr_buf(addrlen);
    int ret = fi_getname(&endpoint->fid, addr_buf.data(), &addrlen);
    if (ret) {
        NIXL_ERROR << "fi_getname failed: " << fi_strerror(-ret);
        return NIXL_ERR_BACKEND;
    }

    address = std::string(addr_buf.data(), addrlen);
    return NIXL_SUCCESS;
}

void nixlOFI_Engine::detectHmemCapabilities(struct fi_info* fi_info,
                                            const std::string& provider_name,
                                            bool& cuda_supported,
                                            bool& ze_supported,
                                            bool& synapseai_supported) {
    if (!fi_info || !(fi_info->domain_attr->mr_mode & FI_MR_HMEM)) {
        NIXL_DEBUG << "Provider " << provider_name << " does not support HMEM";
        cuda_supported = false;
        ze_supported = false;
        synapseai_supported = false;
        return;
    }

    // ofi_rxm;verbs may have limited HMEM support - be conservative
    if (provider_name.find("ofi_rxm") != std::string::npos && 
        provider_name.find("verbs") != std::string::npos) {
        NIXL_DEBUG << "Layered provider " << provider_name << " - validating HMEM support carefully";
    }

    struct {
        const char* name;
        enum fi_hmem_iface iface;
        bool& flag;
    } hmem_checks[] = {
        {"NVIDIA CUDA", FI_HMEM_CUDA, cuda_supported},
        {"Gaudi SynapseAI", FI_HMEM_SYNAPSEAI, synapseai_supported},
        {"Intel Level Zero", FI_HMEM_ZE, ze_supported}
    };

    // use libfabric's HMEM detection for each interface
    for (const auto& check : hmem_checks) {
        struct fi_info *hmem_hints = fi_dupinfo(fi_info);
        struct fi_info *hmem_info = nullptr;

        if (hmem_hints) {
            // test specific HMEM interface support
            hmem_hints->caps |= FI_HMEM;

            int ret = fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0, hmem_hints, &hmem_info);
            if (ret == 0 && hmem_info) {
                // verify the provider actually supports this HMEM interface
                check.flag = (hmem_info->caps & FI_HMEM) != 0;
                if (check.flag) {
                    NIXL_DEBUG << check.name << " HMEM support detected for provider " << provider_name;
                }
                fi_freeinfo(hmem_info);
            } else {
                check.flag = false;
                NIXL_DEBUG << check.name << " HMEM support not available: " << fi_strerror(-ret);
            }
            fi_freeinfo(hmem_hints);
        } else {
            // fallback
            check.flag = false;
            NIXL_WARN << "Failed to duplicate fi_info for " << check.name << " detection";
        }
    }
}
