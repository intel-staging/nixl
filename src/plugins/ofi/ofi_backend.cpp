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
#include <stdexcept>

nixlOfiEngine::nixlOfiEngine(const nixlBackendInitParams* init_params) :
    nixlBackendEngine(init_params),
    fabric_(nullptr),
    domain_(nullptr),
    ep_(nullptr),
    cq_(nullptr),
    eq_(nullptr),
    pep_(nullptr),
    fi_(nullptr),
    cachedProviderInfo_(nullptr),
    av_(nullptr),
    isConnectionless_(false),
    eqThreadStop_(false),
    eqThreadPaused_(false),
    eqTimeoutMs_(100),
    hmemZeSupported_(false),
    hmemCudaSupported_(false),
    hmemSynapseaiSupported_(false)
{
    localAgentName_ = init_params->localAgent;
    struct fi_info *hints = nullptr;
    struct fi_info *info = nullptr;
    int ret = 0;

    // use FI_PROVIDER environment variable or fall back to sensible defaults
    const char* env_provider = getenv("FI_PROVIDER");
    if (env_provider) {
        providerName_ = env_provider;
        NIXL_DEBUG << "Using FI_PROVIDER environment variable: " << providerName_;
    } else {
        // Default to verbs provider
        providerName_ = "verbs";
        NIXL_DEBUG << "Using default provider: " << providerName_;
    }

    // validate that the provider is supported
    const auto* config = findProviderConfig(providerName_);
    if (!config) {
        NIXL_ERROR << "Unsupported provider: " << providerName_;
        NIXL_ERROR << "Supported providers: shm, tcp, verbs";
        this->initErr = true;
        return;
    }

    // get EQ timeout parameter (0-60 seconds max)
    getLongParam(init_params, "eq_timeout_ms", eqTimeoutMs_, 0, 60000);

    hints = fi_allocinfo();
    if (!hints) {
        this->initErr = true;
        NIXL_ERROR << "fi_allocinfo failed";
        return;
    }

    // FI_HMEM for heterogeneous memory support (GPU memory)
    // check if HMEM support is needed via parameter (default: false for DRAM-only usage)
    std::string enable_hmem = "false";
    getStringParam(init_params, "enable_hmem", enable_hmem);
    bool need_hmem = (enable_hmem == "true" || enable_hmem == "1");
    
    // also check environment variable for HMEM enablement
    const char* env_hmem = getenv("NIXL_ENABLE_HMEM");
    if (env_hmem && (std::string(env_hmem) == "true" || std::string(env_hmem) == "1")) {
        need_hmem = true;
        NIXL_INFO << "HMEM explicitly enabled via NIXL_ENABLE_HMEM environment variable";
    }
    
    // auto-enable HMEM if device environments are detected
    const char* cuda_env = getenv("CUDA_VISIBLE_DEVICES");
    const char* habana_env = getenv("HABANA_VISIBLE_DEVICES");  
    const char* ze_env = getenv("ZE_AFFINITY_MASK");
    
    if (habana_env || cuda_env || ze_env) {
        // Only auto-enable HMEM for providers that support it
        const auto* config = findProviderConfig(providerName_);
        if (config && (config->caps & FI_HMEM)) {
            need_hmem = true;
            NIXL_INFO << "HMEM auto-enabled for provider " << providerName_ << " due to device environment detection";
            if (habana_env) NIXL_INFO << "  ... Found HABANA_VISIBLE_DEVICES=" << habana_env;
            if (cuda_env) NIXL_INFO << "  ... Found CUDA_VISIBLE_DEVICES=" << cuda_env;  
            if (ze_env) NIXL_INFO << "  ... Found ZE_AFFINITY_MASK=" << ze_env;
        } else {
            NIXL_INFO << "Device environments detected but provider " << providerName_ << " does not support HMEM";
            if (habana_env) NIXL_INFO << "  ... Found HABANA_VISIBLE_DEVICES=" << habana_env << " (ignored)";
            if (cuda_env) NIXL_INFO << "  ... Found CUDA_VISIBLE_DEVICES=" << cuda_env << " (ignored)";  
            if (ze_env) NIXL_INFO << "  ... Found ZE_AFFINITY_MASK=" << ze_env << " (ignored)";
        }
    }

    NIXL_INFO << "HMEM support requested: " << (need_hmem ? "YES" : "NO");

    if (need_hmem) {
        hints->caps |= FI_HMEM;
        NIXL_INFO << "Adding FI_HMEM to hints->caps for device memory support";
    } else {
        NIXL_INFO << "HMEM not enabled - DRAM memory only";
    }

    // for shm and tcp providers, use minimal configuration (only provider name)
    if (providerName_ == "shm" || providerName_ == "tcp") {
        hints->fabric_attr->prov_name = strdup(providerName_.c_str());
        NIXL_INFO << "Using minimal auto-negotiated hints for " << providerName_ << " provider";
    } else {
        // for other providers, use predefined configuration from SUPPORTED_PROVIDERS  
        configureHintsForProvider(hints, providerName_);
    }

    // debug print all hints
    NIXL_DEBUG << "=== constructor: fi_getinfo hints ===";
    NIXL_DEBUG << "provider name: " << hints->fabric_attr->prov_name;
    NIXL_DEBUG << "caps: " << fi_tostr(&hints->caps, FI_TYPE_CAPS);
    NIXL_DEBUG << "mode: " << fi_tostr(&hints->mode, FI_TYPE_MODE);
    NIXL_DEBUG << "ep_attr->type: " << fi_tostr(&hints->ep_attr->type, FI_TYPE_EP_TYPE);
    NIXL_DEBUG << "domain_attr->mr_mode: " << fi_tostr(&hints->domain_attr->mr_mode, FI_TYPE_MR_MODE);
    NIXL_DEBUG << "domain_attr->resource_mgmt: " << hints->domain_attr->resource_mgmt;
    NIXL_DEBUG << "addr_format: " << fi_tostr(&hints->addr_format, FI_TYPE_ADDR_FORMAT);
    NIXL_DEBUG << "========================";

    // let libfabric choose optimal settings; only override if explicitly needed
    // (removed provider-specific tuning - use environment variables instead)

    ret = fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0, hints, &info);
    if (ret) {
        NIXL_ERROR << "fi_getinfo failed: " << fi_strerror(-ret);
        NIXL_INFO << "Trying fi_getinfo with minimal hints for provider " << providerName_;
        
        // minimal hints, see what provider supports
        struct fi_info *minimal_hints = fi_allocinfo();
        if (minimal_hints) {
            minimal_hints->fabric_attr->prov_name = strdup(providerName_.c_str());
            struct fi_info *minimal_fi = nullptr;
            int minimal_ret = fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0, minimal_hints, &minimal_fi);
            if (minimal_ret == 0) {
                NIXL_INFO << "Provider " << providerName_ << " supports: caps=0x" << std::hex << minimal_fi->caps 
                         << " ep_type=" << minimal_fi->ep_attr->type;
                fi_freeinfo(minimal_fi);
            } else {
                NIXL_ERROR << "Even minimal fi_getinfo failed: " << fi_strerror(-minimal_ret);
            }
            fi_freeinfo(minimal_hints);
        }
        goto cleanup_teardown;
    }

    // use the first provider returned by fi_getinfo (highest performance)
    fi_ = info;
    if (!fi_) {
        NIXL_ERROR << "No providers returned by fi_getinfo";
        goto cleanup_teardown;
    }
    
    NIXL_DEBUG << "Selected provider: " << (fi_->fabric_attr->prov_name ? fi_->fabric_attr->prov_name : "unknown")
               << " with endpoint type: " << fi_tostr(&fi_->ep_attr->type, FI_TYPE_EP_TYPE);

    // connectionless provider?
    isConnectionless_ = isConnectionlessProvider();
    NIXL_DEBUG << "Provider " << providerName_ << " ep_type=" 
               << fi_tostr(&fi_->ep_attr->type, FI_TYPE_EP_TYPE) 
               << " isConnectionless=" << isConnectionless_;

    // detect HMEM capabilities for this provider
    detectHmemCapabilities(fi_, providerName_, hmemCudaSupported_,
                           hmemZeSupported_, hmemSynapseaiSupported_);

    ret = fi_fabric(fi_->fabric_attr, &fabric_, nullptr);
    if (ret) {
        NIXL_ERROR << "fi_fabric failed: " << fi_strerror(-ret);
        goto cleanup_teardown;
    }

    ret = fi_domain(fabric_, fi_, &domain_, nullptr);
    if (ret) {
        NIXL_ERROR << "fi_domain failed: " << fi_strerror(-ret);
        goto cleanup_teardown;
    }

    if (setupEndpoint(!isConnectionless_) != NIXL_SUCCESS) {
        goto cleanup_teardown;
    }

    // get local address
    if (getEndpointAddress(ep_, localAddr_) != NIXL_SUCCESS) {
        goto cleanup_teardown;
    }

    // cache provider use in connect()
    cachedProviderInfo_ = fi_dupinfo(fi_);
    if (!cachedProviderInfo_) {
        NIXL_WARN << "Failed to duplicate provider info for caching";
    }

    fi_freeinfo(hints);
    fi_freeinfo(info);

    // start event loop thread for connection-oriented providers
    if (!isConnectionless_) {
        eqThread_ = std::thread(&nixlOfiEngine::eq_event_loop, this);
    }
    return;

cleanup_teardown:
    if (ep_)     { fi_close(&ep_->fid);     ep_ = nullptr; }
    if (av_)     { fi_close(&av_->fid);     av_ = nullptr; }
    if (pep_)    { fi_close(&pep_->fid);    pep_ = nullptr; }
    if (eq_)     { fi_close(&eq_->fid);     eq_ = nullptr; }
    if (cq_)     { fi_close(&cq_->fid);     cq_ = nullptr; }
    if (domain_) { fi_close(&domain_->fid); domain_ = nullptr; }
    if (fabric_) { fi_close(&fabric_->fid); fabric_ = nullptr; }
    if (info)    { fi_freeinfo(info);       info = nullptr; }
    if (hints)   { fi_freeinfo(hints);      hints = nullptr; }
    this->initErr = true;
}

// Parameter extraction helper methods
void nixlOfiEngine::getStringParam(const nixlBackendInitParams* init_params, const std::string& key, std::string& value) {
    auto it = init_params->customParams->find(key);
    if (it != init_params->customParams->end()) {
        value = it->second;
    }
}

void nixlOfiEngine::getLongParam(const nixlBackendInitParams* init_params, const std::string& key, long& value, long min_val, long max_val) {
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

void nixlOfiEngine::getSizeTParam(const nixlBackendInitParams* init_params, const std::string& key, size_t& value) {
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

// Predefined provider configurations for providers that need explicit settings
// Note: SHM and TCP providers use minimal auto-negotiated configuration instead
const nixlOfiEngine::ProviderConfig nixlOfiEngine::SUPPORTED_PROVIDERS[] = {
    {   
        "shm",
        FI_EP_RDM,
        FI_HMEM, // not implemented Gaudi HBM
        0,  // let provider choose mode
        0,  // let provider choose MR mode  
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
        // https://github.com/ofiwg/libfabric/wiki/Provider-Feature-Matrix-main
        "verbs",
        FI_EP_RDM,
        FI_MSG | FI_HMEM | FI_READ | FI_WRITE, // match fabtests exactly - only FI_MSG capability
        0,
        FI_MR_LOCAL | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_ENDPOINT | FI_MR_HMEM, // match fabtests mr_mode 636
        FI_RM_ENABLED,
        {0, 0, 0, 0, 0, 0, 0, 0, FI_TC_BULK_DATA}, // tx_attr with bulk data class like fabtests
        {0, 0, 0, 0, 0, 0}, // rx_attr defaults
        FI_FORMAT_UNSPEC,
        FI_PROGRESS_AUTO,
        FI_PROGRESS_AUTO
    }
};

const size_t nixlOfiEngine::NUM_SUPPORTED_PROVIDERS =
    sizeof(SUPPORTED_PROVIDERS) / sizeof(SUPPORTED_PROVIDERS[0]);

const nixlOfiEngine::ProviderConfig* nixlOfiEngine::findProviderConfig(const std::string& provider_name) {
    for (size_t i = 0; i < NUM_SUPPORTED_PROVIDERS; ++i) {
        if (SUPPORTED_PROVIDERS[i].name == provider_name) {
            return &SUPPORTED_PROVIDERS[i];
        }
    }
    return nullptr;
}

void nixlOfiEngine::configureHintsForProvider(struct fi_info* hints, const std::string& provider_name) {
    const auto* config = findProviderConfig(provider_name);

    if (!config) {
        // if the provider is not in our list, use the verbs config as a safe default
        config = findProviderConfig("verbs");
        NIXL_DEBUG << "Unknown provider '" << provider_name << "', using verbs config as a fallback.";
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
    // but only for connection-oriented (FI_EP_MSG) endpoints, not RDM
    if (provider_name.find("verbs") != std::string::npos && config->ep_type == FI_EP_MSG) {
        hints->ep_attr->rx_ctx_cnt = FI_SHARED_CONTEXT;
    }

    // always set the provider name in the hints
    if (hints->fabric_attr->prov_name) free(hints->fabric_attr->prov_name);
    hints->fabric_attr->prov_name = strdup(provider_name.c_str());
}

nixlOfiEngine::~nixlOfiEngine() {
    if (!isConnectionless_) {
        eqThreadStop_ = true;
        if (eqThread_.joinable()) {
            // wake up the EQ thread to ensure it exits
            if (eq_) {
                uint32_t event;
                fi_eq_read(eq_, &event, nullptr, 0, 0);
            }
            eqThread_.join();
        }
    }

    // close connected endpoints
    for (auto const& [key, val] : connectedEps_) {
        fi_close(&val->fid);
    }

    if (pep_)    { fi_close(&pep_->fid);    pep_ = nullptr; }
    if (ep_)     { fi_close(&ep_->fid);     ep_ = nullptr; }
    if (cq_)     { fi_close(&cq_->fid);     cq_ = nullptr; }
    if (eq_)     { fi_close(&eq_->fid);     eq_ = nullptr; }
    if (av_)     { fi_close(&av_->fid);     av_ = nullptr; }
    if (domain_) { fi_close(&domain_->fid); domain_ = nullptr; }
    if (fabric_) { fi_close(&fabric_->fid); fabric_ = nullptr; }
    if (cachedProviderInfo_) fi_freeinfo(cachedProviderInfo_);
}

bool nixlOfiEngine::supportsNotif() const {
    return false;
}

bool nixlOfiEngine::supportsRemote() const {
    return true;
}

bool nixlOfiEngine::supportsLocal() const {
    return false;
}

bool nixlOfiEngine::supportsProgTh() const {
    return true;
}

nixl_mem_list_t nixlOfiEngine::getSupportedMems() const {
    nixl_mem_list_t mems;
    mems.push_back(DRAM_SEG);
    if (hmemCudaSupported_) {
        mems.push_back(VRAM_SEG);
    }
    if (hmemSynapseaiSupported_) {
        mems.push_back(VRAM_SEG);
    }
    if (hmemZeSupported_) {
        mems.push_back(VRAM_SEG);
    }
    return mems;
}

nixl_status_t nixlOfiEngine::connect(const std::string &remote_agent) {
    NIXL_DEBUG << "connect() called for remote_agent: " << remote_agent 
               << " isConnectionless: " << isConnectionless_;
    std::lock_guard<std::mutex> lock(epLock_);

    if (isConnectionless_) {
        // for connectionless providers like shm: insert remote address into av
        if (shmAddrs_.count(remote_agent)) {
            NIXL_DEBUG << "Already have address mapping for " << remote_agent;
            return NIXL_SUCCESS;
        }

        auto remote_addr_it = remoteAddrs_.find(remote_agent);
        if (remote_addr_it == remoteAddrs_.end()) {
            NIXL_ERROR << "Remote address for " << remote_agent << " not found.";
            return NIXL_ERR_NOT_FOUND;
        }

        fi_addr_t addr;
        int ret = fi_av_insert(av_, remote_addr_it->second.data(), 1, &addr, 0, nullptr);
        if (ret != 1) {
            NIXL_ERROR << "fi_av_insert failed: " << fi_strerror(-ret);
            return NIXL_ERR_BACKEND;
        }

        shmAddrs_[remote_agent] = addr;
        NIXL_DEBUG << "OFI backend: Added address mapping for " << remote_agent;
        return NIXL_SUCCESS;
    }

    // connection-oriented logic
    if (connectedEps_.count(remote_agent)) {
        NIXL_DEBUG << "Already connected to " << remote_agent;
        return NIXL_SUCCESS;
    }

    auto remote_addr_it = remoteAddrs_.find(remote_agent);
    if (remote_addr_it == remoteAddrs_.end()) {
        NIXL_ERROR << "Remote address for " << remote_agent << " not found.";
        return NIXL_ERR_NOT_FOUND;
    }
    const std::string &remote_addr_str = remote_addr_it->second;

    // create copy of provider info to avoid shared state issues
    struct fi_info *remote_fi = fi_dupinfo(cachedProviderInfo_);
    if (!remote_fi) {
        NIXL_ERROR << "Failed to duplicate provider info for remote agent";
        return NIXL_ERR_BACKEND;
    }

    // update dest_addr for this connection
    remote_fi->dest_addr = (void*)remote_addr_str.c_str();
    remote_fi->dest_addrlen = remote_addr_str.length();
    
    
    // let libfabric use the auto-negotiated address format from fi_getinfo

    fid_ep *remote_ep = nullptr;
    int ret = fi_endpoint(domain_, remote_fi, &remote_ep, nullptr);
    if (ret) {
        NIXL_ERROR << "fi_endpoint for remote failed: " << fi_strerror(-ret);
        fi_freeinfo(remote_fi);
        return NIXL_ERR_BACKEND;
    }

    ret = fi_ep_bind(remote_ep, &cq_->fid, FI_SEND | FI_RECV);
    if (ret) {
        NIXL_ERROR << "fi_ep_bind to CQ for remote failed: " << fi_strerror(-ret);
        fi_close(&remote_ep->fid);
        fi_freeinfo(remote_fi);
        return NIXL_ERR_BACKEND;
    }

    // use no flags for all providers to avoid compatibility issues
    ret = fi_ep_bind(remote_ep, &eq_->fid, 0);
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

    // pause the event loop to prevent it from consuming our FI_CONNECTED event
    eqThreadPaused_.store(true);
    // Give the event loop time to notice the pause and stop processing
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    ret = fi_connect(remote_ep, remote_fi->dest_addr, localAgentName_.c_str(), localAgentName_.length() + 1);
    if (ret) {
        NIXL_ERROR << "fi_connect failed: " << fi_strerror(-ret);
        // resume event loop before returning
        eqThreadPaused_.store(false);
        eqPauseCV_.notify_one();
        fi_close(&remote_ep->fid);
        fi_freeinfo(remote_fi);
        return NIXL_ERR_BACKEND;
    }

    // wait for connection to complete via EQ
    // use fi_eq_sread (similar to fabtests implementation)
    // for blocking synchronous read to avoid race with event loop
    struct fi_eq_cm_entry entry;
    uint32_t event;
    ssize_t n_events = fi_eq_sread(eq_, &event, &entry, sizeof(entry), -1, 0);
    
    // resume event loop now that we got our event
    eqThreadPaused_.store(false);
    eqPauseCV_.notify_one();
    
    if (n_events != sizeof(entry)) {
        if (n_events < 0) {
            NIXL_ERROR << "fi_eq_sread failed during connect: " << fi_strerror(-n_events);
            // Try to read error details if available
            if (n_events == -FI_EAVAIL) {
                struct fi_eq_err_entry err_entry;
                ssize_t err_ret = fi_eq_readerr(eq_, &err_entry, 0);
                if (err_ret == sizeof(err_entry)) {
                    NIXL_ERROR << "EQ error details: prov_errno=" << err_entry.prov_errno 
                               << " err=" << err_entry.err << " (" << fi_strerror(err_entry.err) << ")";
                }
            }
        } else {
            NIXL_ERROR << "fi_eq_sread returned unexpected size: " << n_events << " (expected " << sizeof(entry) << ")";
        }
        fi_close(&remote_ep->fid);
        fi_freeinfo(remote_fi);
        return NIXL_ERR_BACKEND;
    }
    if (event != FI_CONNECTED || entry.fid != &remote_ep->fid) {
        NIXL_ERROR << "Unexpected EQ event during connect: " << event << " (expected FI_CONNECTED=" << FI_CONNECTED << ")";
        fi_close(&remote_ep->fid);
        fi_freeinfo(remote_fi);
        return NIXL_ERR_BACKEND;
    }

    connectedEps_[remote_agent] = remote_ep;
    fi_freeinfo(remote_fi);

    NIXL_DEBUG << "OFI backend: Connected to " << remote_agent;
    return NIXL_SUCCESS;
}

nixl_status_t nixlOfiEngine::disconnect(const std::string &remote_agent) {
    std::lock_guard<std::mutex> lock(epLock_);

    if (isConnectionless_) {
        // connectionless provider, remove address mapping
        auto it = shmAddrs_.find(remote_agent);
        if (it == shmAddrs_.end()) {
            NIXL_WARN << "OFI backend: No address mapping for " << remote_agent;
            return NIXL_ERR_NOT_FOUND;
        }

        int ret = fi_av_remove(av_, &it->second, 1, 0);
        if (ret) {
            NIXL_ERROR << "fi_av_remove failed: " << fi_strerror(-ret);
            return NIXL_ERR_BACKEND;
        }

        shmAddrs_.erase(it);
        NIXL_DEBUG << "OFI backend: Removed address mapping for " << remote_agent;
        return NIXL_SUCCESS;
    }

    // connection-oriented case
    auto it = connectedEps_.find(remote_agent);
    if (it == connectedEps_.end()) {
        NIXL_WARN << "OFI backend: No active connection to " << remote_agent;
        return NIXL_ERR_NOT_FOUND;
    }

    // Store endpoint before erasing to ensure proper cleanup even if fi_close fails
    fid_ep* ep_to_close = it->second;
    connectedEps_.erase(it);
    
    int ret = fi_close(&ep_to_close->fid);
    if (ret) {
        NIXL_ERROR << "fi_close (remote_ep) failed: " << fi_strerror(-ret);
        return NIXL_ERR_BACKEND;
    }
    NIXL_DEBUG << "OFI backend: Disconnected from " << remote_agent;
    return NIXL_SUCCESS;
}

nixl_status_t nixlOfiEngine::registerMem(const nixlBlobDesc &mem,
                                     const nixl_mem_t &nixl_mem,
                                     nixlBackendMD* &out) {
    nixlOfiMetadata *ofi_meta = new nixlOfiMetadata();
    if (!ofi_meta) {
        return NIXL_ERR_BACKEND;
    }

    nixl_status_t status = NIXL_SUCCESS;
    
    if (nixl_mem == DRAM_SEG) {
        status = registerDramMemory(mem, ofi_meta);
    } else if (nixl_mem == VRAM_SEG) {
        status = registerVramMemory(mem, ofi_meta);
    } else {
        NIXL_ERROR << "Unsupported memory type: " << nixl_mem;
        delete ofi_meta;
        return NIXL_ERR_NOT_SUPPORTED;
    }

    if (status != NIXL_SUCCESS) {
        delete ofi_meta;
        return status;
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

nixl_status_t nixlOfiEngine::deregisterMem(nixlBackendMD *meta) {
    nixlOfiMetadata *ofi_meta = static_cast<nixlOfiMetadata*>(meta);
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

nixl_status_t nixlOfiEngine::unloadMD(nixlBackendMD* input) {
    return deregisterMem(input);
}

nixl_status_t nixlOfiEngine::prepXfer(const nixl_xfer_op_t &operation,
                                  const nixl_meta_dlist_t &local,
                                  const nixl_meta_dlist_t &remote,
                                  const std::string &remote_agent,
                                  nixlBackendReqH* &handle,
                                  const nixl_opt_b_args_t* opt_args) const {
    return postXfer(operation, local, remote, remote_agent, handle, opt_args);
}

nixl_status_t nixlOfiEngine::postXfer(const nixl_xfer_op_t &operation,
                                  const nixl_meta_dlist_t &local,
                                  const nixl_meta_dlist_t &remote,
                                  const std::string &remote_agent,
                                  nixlBackendReqH* &handle,
                                  const nixl_opt_b_args_t* opt_args) const {
    fid_ep *target_ep = ep_;
    fi_addr_t dest_addr = FI_ADDR_UNSPEC;

    if (isConnectionless_) {
        auto shm_it = shmAddrs_.find(remote_agent);
        if (shm_it == shmAddrs_.end()) {
            NIXL_ERROR << "OFI backend: No address mapping for " << remote_agent;
            return NIXL_ERR_NOT_FOUND;
        }
        dest_addr = shm_it->second;
    } else {
        auto it = connectedEps_.find(remote_agent);
        if (it == connectedEps_.end()) {
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

    nixlOfiRequest *ofi_req = new nixlOfiRequest();
    if (!ofi_req) {
        return NIXL_ERR_BACKEND;
    }
    ofi_req->cq = cq_;

    int ret = 0;
    for (size_t i = 0; i < static_cast<size_t>(local.descCount()); ++i) {
        const nixlMetaDesc &local_desc = local[i];
        const nixlMetaDesc &remote_desc = remote[i];

        nixlOfiMetadata *local_meta = static_cast<nixlOfiMetadata*>(local_desc.metadataP);
        nixlOfiMetadata *remote_meta = static_cast<nixlOfiMetadata*>(remote_desc.metadataP);
        
        if (!local_meta || !remote_meta || !local_meta->mr) {
            NIXL_ERROR << "Invalid metadata or memory registration";
            delete ofi_req;
            return NIXL_ERR_INVALID_PARAM;
        }

        // get remote memory key - either from mr or stored in desc field for remote metadata
        uint64_t remote_key;
        if (remote_meta->mr) {
            remote_key = fi_mr_key(remote_meta->mr);
        } else {
            // for remote metadata, key is stored in desc field
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
            if (ret == -FI_EAGAIN) {
                // RDM provider: FI_EAGAIN means operation was posted successfully
                NIXL_DEBUG << "OFI transfer posted asynchronously (EAGAIN) - will complete via polling";
                // continue to success path - operation posted successfully, will complete async
                break;
            } else {
                NIXL_ERROR << "OFI transfer failed: " << fi_strerror(-ret);
                delete ofi_req;
                return NIXL_ERR_BACKEND;
            }
        }
    }

    handle = ofi_req;
    return NIXL_SUCCESS;

}

nixl_status_t nixlOfiEngine::checkXfer(nixlBackendReqH* handle) const {
    nixlOfiRequest *ofi_req = static_cast<nixlOfiRequest*>(handle);
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

nixl_status_t nixlOfiEngine::releaseReqH(nixlBackendReqH* handle) const {
    nixlOfiRequest *ofi_req = static_cast<nixlOfiRequest*>(handle);
    if (!ofi_req) {
        return NIXL_ERR_INVALID_PARAM;
    }
    delete ofi_req;
    return NIXL_SUCCESS;
}

nixl_status_t nixlOfiEngine::getConnInfo(std::string &conn_info) const {
    conn_info = localAddr_;
    return NIXL_SUCCESS;
}

nixl_status_t nixlOfiEngine::loadRemoteConnInfo(const std::string &remote_agent, const std::string &conn_info) {
    std::lock_guard<std::mutex> lock(epLock_);
    remoteAddrs_[remote_agent] = conn_info;
    return NIXL_SUCCESS;
}

nixl_status_t nixlOfiEngine::getPublicData(const nixlBackendMD* meta, std::string &str) const {
    const nixlOfiMetadata* ofi_meta = static_cast<const nixlOfiMetadata*>(meta);
    if (!ofi_meta || !ofi_meta->mr) {
        return NIXL_ERR_INVALID_PARAM;
    }
    
    // serialize memory registration key for remote access
    uint64_t mr_key = fi_mr_key(ofi_meta->mr);
    str = std::to_string(mr_key);
    return NIXL_SUCCESS;
}

nixl_status_t nixlOfiEngine::loadRemoteMD(const nixlBlobDesc &input, const nixl_mem_t &nixl_mem,
                                           const std::string &remote_agent, nixlBackendMD* &output) {
    // create a remote metadata object from the serialized public data
    nixlOfiMetadata* remote_meta = new nixlOfiMetadata();
    if (!remote_meta) {
        return NIXL_ERR_BACKEND;
    }
    
    // parse the memory key from the metadata string
    try {
        // the metaInfo should contain the memory key as a string
        std::string key_str(input.metaInfo.data(), input.metaInfo.size());
        uint64_t remote_key = std::stoull(key_str);
        
        // for remote metadata, we don't have an actual mr object, just the key
        // store the key for later use in RMA operations
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

void nixlOfiEngine::eq_event_loop() {
    while (!eqThreadStop_) {
        // check if we need to pause the event loop during client connections
        if (eqThreadPaused_.load()) {
            std::unique_lock<std::mutex> lock(eqPauseMutex_);
            eqPauseCV_.wait(lock, [this] { return !eqThreadPaused_.load() || eqThreadStop_.load(); });
            if (eqThreadStop_) {
                break;
            }
        }
        
        struct fi_eq_cm_entry entry;
        uint32_t event;
        ssize_t ret = fi_eq_read(eq_, &event, &entry, 1, eqTimeoutMs_);

        if (ret == -FI_EAGAIN) {
            continue;
        } else if (ret < 0) {
            if (ret == -FI_EINTR && eqThreadStop_) {
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
                int connreq_ret = fi_endpoint(domain_, fi_, &new_ep, nullptr);
                if (connreq_ret) {
                    NIXL_ERROR << "fi_endpoint for accepted connection failed: " << fi_strerror(-connreq_ret);
                    break;
                }
                connreq_ret = fi_ep_bind(new_ep, &cq_->fid, FI_SEND | FI_RECV);
                if (connreq_ret) {
                    NIXL_ERROR << "fi_ep_bind to CQ for accepted connection failed: " << fi_strerror(-connreq_ret);
                    fi_close(&new_ep->fid);
                    break;
                }
                // use no flags for EQ binding to avoid compatibility issues
                connreq_ret = fi_ep_bind(new_ep, &eq_->fid, 0);
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

                std::lock_guard<std::mutex> lock(epLock_);
                connectedEps_[remote_agent_name] = new_ep;
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
                    std::lock_guard<std::mutex> lock(epLock_);
                    for (auto it = connectedEps_.begin(); it != connectedEps_.end(); ++it) {
                        if (&it->second->fid == entry.fid) {
                            fi_close(&it->second->fid);
                            connectedEps_.erase(it);
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

bool nixlOfiEngine::isConnectionlessProvider() const {
    // use libfabric's endpoint type to determine connection model
    // FI_EP_RDM (Reliable Datagram) = connectionless
    // FI_EP_MSG (Message) = connection-oriented  
    // FI_EP_DGRAM (Datagram) = connectionless
    if (fi_ && fi_->ep_attr) {
        return (fi_->ep_attr->type == FI_EP_RDM || fi_->ep_attr->type == FI_EP_DGRAM);
    }
    
    // fallback: if fi_ not available yet, use provider names for known cases
    if (providerName_.find("ofi_rxm") != std::string::npos || 
        providerName_ == "shm" || providerName_ == "udp") {
        return true;
    }
    
    return false;
}

nixl_status_t nixlOfiEngine::setupEndpoint(bool connection_oriented) {
    int ret = 0;

    // create endpoint
    ret = fi_endpoint(domain_, fi_, &ep_, nullptr);
    if (ret) {
        NIXL_ERROR << "fi_endpoint failed: " << fi_strerror(-ret);
        return NIXL_ERR_BACKEND; // ep_ was never created, no cleanup needed
    }

    // create and bind completion queue
    struct fi_cq_attr cq_attr = {};
    cq_attr.size = 128; // use fi_->tx_attr->size + fi_->rx_attr->size?
    cq_attr.format = FI_CQ_FORMAT_CONTEXT;
    ret = fi_cq_open(domain_, &cq_attr, &cq_, nullptr);
    if (ret) {
        NIXL_ERROR << "fi_cq_open failed: " << fi_strerror(-ret);
        goto cleanup_setup;
    }

    ret = fi_ep_bind(ep_, &cq_->fid, FI_SEND | FI_RECV);
    if (ret) {
        NIXL_ERROR << "fi_ep_bind to CQ failed: " << fi_strerror(-ret);
        goto cleanup_setup;
    }

    if (connection_oriented) {
        // event queue for connection management
        struct fi_eq_attr eq_attr = {};
        eq_attr.size = 64;
        eq_attr.wait_obj = FI_WAIT_UNSPEC;
        ret = fi_eq_open(fabric_, &eq_attr, &eq_, nullptr);
        if (ret) {
            NIXL_ERROR << "fi_eq_open failed: " << fi_strerror(-ret);
            goto cleanup_setup;
        }

        // bind endpoint to EQ for connection management
        ret = fi_ep_bind(ep_, &eq_->fid, 0);
        if (ret) {
            NIXL_ERROR << "fi_ep_bind to EQ failed: " << fi_strerror(-ret);
            goto cleanup_setup;
        }

        // create passive endpoint for listening
        ret = fi_passive_ep(fabric_, fi_, &pep_, nullptr);
        if (ret) {
            const char* prov = (fi_ && fi_->fabric_attr && fi_->fabric_attr->prov_name)
                               ? fi_->fabric_attr->prov_name : "unknown";
            NIXL_ERROR << "fi_passive_ep failed on provider=" << prov
                       << " ep_type=" << fi_tostr(&fi_->ep_attr->type, FI_TYPE_EP_TYPE)
                       << " err=" << -ret << " (" << fi_strerror(-ret) << ")";
            goto cleanup_setup;
        }

        ret = fi_pep_bind(pep_, &eq_->fid, 0);
        if (ret) {
            NIXL_ERROR << "fi_pep_bind to EQ failed: " << fi_strerror(-ret);
            goto cleanup_setup;
        }

        ret = fi_listen(pep_);
        if (ret) {
            NIXL_ERROR << "fi_listen failed: " << fi_strerror(-ret);
            goto cleanup_setup;
        }
        NIXL_INFO << "TCP passive endpoint listening for connections";
    } else {
        // address vector for connectionless communication
        struct fi_av_attr av_attr = {};
        av_attr.type = FI_AV_MAP;
        ret = fi_av_open(domain_, &av_attr, &av_, nullptr);
        if (ret) {
            NIXL_ERROR << "fi_av_open failed: " << fi_strerror(-ret);
            goto cleanup_setup;
        }

        ret = fi_ep_bind(ep_, &av_->fid, 0);
        if (ret) {
            NIXL_ERROR << "fi_ep_bind to AV failed: " << fi_strerror(-ret);
            goto cleanup_setup;
        }

        ret = fi_enable(ep_);
        if (ret) {
            NIXL_ERROR << "fi_enable failed: " << fi_strerror(-ret);
            goto cleanup_setup;
        }
    }
    return NIXL_SUCCESS;

cleanup_setup:
    // cleanup only what setupEndpoint created, set pointers to nullptr
    // Close endpoint BEFORE CQ since EP depends on CQ
    if (ep_)  { fi_close(&ep_->fid);  ep_ = nullptr; }
    if (av_)  { fi_close(&av_->fid);  av_ = nullptr; }
    if (pep_) { fi_close(&pep_->fid); pep_ = nullptr; }
    if (eq_)  { fi_close(&eq_->fid);  eq_ = nullptr; }
    if (cq_)  { fi_close(&cq_->fid);  cq_ = nullptr; }
    return NIXL_ERR_BACKEND;
}

nixl_status_t nixlOfiEngine::getEndpointAddress(fid_ep* endpoint, std::string& address) {
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

void nixlOfiEngine::detectHmemCapabilities(struct fi_info* fi_info,
                                            const std::string& provider_name,
                                            bool& cuda_supported,
                                            bool& ze_supported,
                                            bool& synapseai_supported) {
    // Check if provider supports generic HMEM capability
    if (!fi_info || !(fi_info->caps & FI_HMEM)) {
        NIXL_DEBUG << "Provider " << provider_name << " does not support HMEM";
        cuda_supported = false;
        ze_supported = false;
        synapseai_supported = false;
        return;
    }

    NIXL_DEBUG << "Provider " << provider_name << " supports generic HMEM capability";

    // for now, conservatively enable all interfaces if HMEM is supported
    // TODO: determine which specific interfaces actually work
    cuda_supported = true;
    ze_supported = true; 
    synapseai_supported = true;

    NIXL_DEBUG << "HMEM interfaces marked as potentially available - runtime detection will validate";
}

uint64_t nixlOfiEngine::getMemoryRegistrationAccessFlags(const struct fi_info* fi_info) {
    uint64_t access_flags = FI_REMOTE_READ | FI_REMOTE_WRITE | FI_SEND | FI_RECV;
    
    if (fi_info && fi_info->domain_attr) {
        if (fi_info->caps & FI_READ) access_flags |= FI_READ;
        if (fi_info->caps & FI_WRITE) access_flags |= FI_WRITE;
        if (fi_info->caps & FI_RMA) {
            access_flags |= FI_READ | FI_WRITE;
        }
    }
    
    return access_flags;
}

fi_hmem_iface nixlOfiEngine::selectHmemInterface(const nixlBlobDesc &mem, uint64_t &device_id) const {
    device_id = mem.devId >= 0 ? mem.devId : 0;
    
    const char* cuda_env = getenv("CUDA_VISIBLE_DEVICES");
    const char* habana_env = getenv("HABANA_VISIBLE_DEVICES");  
    const char* ze_env = getenv("ZE_AFFINITY_MASK");
    
    // Prefer environment-based detection
    if (habana_env && hmemSynapseaiSupported_) {
        NIXL_INFO << "Using SynapseAI HMEM interface for device " << device_id << " (via HABANA_VISIBLE_DEVICES)";
        return FI_HMEM_SYNAPSEAI;
    }
    if (cuda_env && hmemCudaSupported_) {
        NIXL_INFO << "Using CUDA HMEM interface for device " << device_id << " (via CUDA_VISIBLE_DEVICES)";
        return FI_HMEM_CUDA;
    }
    if (ze_env && hmemZeSupported_) {
        NIXL_INFO << "Using ZE HMEM interface for device " << device_id << " (via ZE_AFFINITY_MASK)";
        return FI_HMEM_ZE;
    }
    
    // Auto-detect based on supported interfaces
    NIXL_INFO << "No HMEM environment variables detected, auto-selecting interface for VRAM";
    if (hmemSynapseaiSupported_) {
        NIXL_INFO << "Auto-selected SynapseAI HMEM interface for device " << device_id;
        return FI_HMEM_SYNAPSEAI;
    }
    if (hmemCudaSupported_) {
        NIXL_INFO << "Auto-selected CUDA HMEM interface for device " << device_id;
        return FI_HMEM_CUDA;
    }
    if (hmemZeSupported_) {
        NIXL_INFO << "Auto-selected ZE HMEM interface for device " << device_id;
        return FI_HMEM_ZE;
    }
    
    NIXL_WARN << "No HMEM interfaces supported. Falling back to host memory registration";
    return FI_HMEM_SYSTEM;
}

nixl_status_t nixlOfiEngine::registerDramMemory(const nixlBlobDesc &mem, nixlOfiMetadata *ofi_meta) const {
    uint64_t access_flags = getMemoryRegistrationAccessFlags(fi_);
    
    int ret = fi_mr_reg(domain_, reinterpret_cast<void*>(mem.addr), mem.len,
                       access_flags, 0, 0, 0, &ofi_meta->mr, nullptr);
    
    if (ret) {
        NIXL_ERROR << "fi_mr_reg failed for DRAM: " << fi_strerror(-ret);
        return NIXL_ERR_BACKEND;
    }
    
    return NIXL_SUCCESS;
}

nixl_status_t nixlOfiEngine::registerVramMemory(const nixlBlobDesc &mem, nixlOfiMetadata *ofi_meta) const {
    struct fi_mr_attr mr_attr = {};
    struct iovec iov = {};
    
    iov.iov_base = reinterpret_cast<void*>(mem.addr);
    iov.iov_len = mem.len;
    
    mr_attr.mr_iov = &iov;
    mr_attr.iov_count = 1;
    mr_attr.access = getMemoryRegistrationAccessFlags(fi_);
    
    uint64_t device_id = 0;
    mr_attr.iface = selectHmemInterface(mem, device_id);
    
    switch (mr_attr.iface) {
        case FI_HMEM_CUDA:
            mr_attr.device.cuda = device_id;
            break;
        case FI_HMEM_ZE:
            mr_attr.device.ze = device_id;
            break;
        case FI_HMEM_SYNAPSEAI:
            mr_attr.device.synapseai = device_id;
            break;
        default:
            mr_attr.device.reserved = 0;
            break;
    }
    
    uint64_t reg_flags = 0;
    if (fi_ && fi_->domain_attr && fi_->domain_attr->mr_mode) {
        if (fi_->domain_attr->mr_mode & FI_MR_HMEM) {
            reg_flags |= FI_HMEM_DEVICE_ONLY;
        }
        NIXL_DEBUG << "Provider MR mode: 0x" << std::hex << fi_->domain_attr->mr_mode 
                  << " using reg_flags: 0x" << reg_flags;
    }
    
    NIXL_DEBUG << "Registering VRAM memory with interface " << mr_attr.iface 
              << " access 0x" << std::hex << mr_attr.access 
              << " flags 0x" << reg_flags;
              
    int ret = fi_mr_regattr(domain_, &mr_attr, reg_flags, &ofi_meta->mr);
    
    if (ret) {
        if (mr_attr.iface == FI_HMEM_SYNAPSEAI) {
            NIXL_ERROR << "SynapseAI device memory registration failed: " << fi_strerror(-ret);
            NIXL_ERROR << "  SynapseAI device is busy or inaccessible.";
            NIXL_ERROR << "  unset HABANA_VISIBLE_DEVICES to use host memory only.";
        } else {
            NIXL_ERROR << "fi_mr_regattr failed: " << fi_strerror(-ret);
        }
        return NIXL_ERR_BACKEND;
    }
    
    return NIXL_SUCCESS;
}

