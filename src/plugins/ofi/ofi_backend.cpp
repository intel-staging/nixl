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
#include <unistd.h>
#include <functional>
#include <fcntl.h>

#include <cstdlib>

// OFI_POST macro based on libfabric FT_POST pattern for reliable operation posting
#define OFI_POST(post_fn, cq, seq, op_str, ...)                             \
    do {                                                                    \
        int ret, progress_ret;                                              \
        while (1) {                                                         \
            ret = post_fn(__VA_ARGS__);                                     \
            if (!ret) {                                                     \
                break;                                                      \
            }                                                               \
            if (ret != -FI_EAGAIN) {                                        \
                NIXL_ERROR << "OFI " op_str " failed: " << fi_strerror(-ret) << " (" << ret << ")"; \
                return NIXL_ERR_BACKEND;                                    \
            }                                                               \
            progress_ret = ofi_progress_manual(cq);                         \
            if (progress_ret < 0 && progress_ret != -FI_EAGAIN) {           \
                NIXL_ERROR << "OFI progress failed during " op_str ": " << fi_strerror(-progress_ret); \
                return NIXL_ERR_BACKEND;                                    \
            }                                                               \
        }                                                                   \
        seq++;                                                              \
    } while (0)

// static synapseAI handles for dynamic loading
void* nixlOfiEngine::synapseai_handle_ = nullptr;
void* nixlOfiEngine::hlthunk_handle_ = nullptr;
nixlOfiEngine::synapseai_ops nixlOfiEngine::synapseai_ops_ = {};
static std::mutex synapseai_init_mutex_;

// progress rate limiting constant
const std::chrono::milliseconds nixlOfiEngine::PROGRESS_INTERVAL{1};

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
    connectionProgressStop_(false),
    shutdownFlag_(false),
    connectionProgressEnabled_(false),
    connectionProgressDelay_(10000), // 10ms for connection events
    lastProgressTime_{std::chrono::steady_clock::now()},
    hmemZeSupported_(false),
    hmemCudaSupported_(false),
    hmemSynapseaiSupported_(false)
{
    int ret = 0;
    struct fi_info *info = nullptr;
    struct fi_info *hints = nullptr;
    localAgentName_ = init_params->localAgent;

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

    // HMEM determination strategy:
    // 1 environment variables: HMEM=1 (auto-detect), HMEM_SYNAPSEAI=1, HMEM_CUDA=1, HMEM_ZE=1
    // 2 application registers VRAM memory type during registerMem() call
    // 3 device discovery during registration if not overridden
    
    bool need_hmem = false;
    
    // check HMEM environment variables
    auto isEnvTrue = [](const char* env_val) -> bool {
        return env_val && (strcmp(env_val, "1") == 0 || strcmp(env_val, "true") == 0);
    };
    
    const char* hmem_vars[] = {"HMEM_SYNAPSEAI", "HMEM_CUDA", "HMEM_ZE", "HMEM"};
    const char* hmem_names[] = {"SynapseAI", "CUDA", "ZE", "auto-detection"};
    
    for (size_t i = 0; i < 4; ++i) {
        const char* env_val = getenv(hmem_vars[i]);
        if (isEnvTrue(env_val)) {
            need_hmem = true;
            NIXL_DEBUG << "HMEM forced to " << hmem_names[i] << " via " << hmem_vars[i] << " environment variable";
            break;
        }
    }

    if (need_hmem) {
        hints->caps |= FI_HMEM;
        NIXL_DEBUG << "Adding FI_HMEM to hints->caps for device memory support";
    } else {
        NIXL_DEBUG << "HMEM not enabled - DRAM memory only";
    }

    // for shm and tcp providers, use minimal configuration (only provider name)
    if (providerName_ == "shm" || providerName_ == "tcp") {
        hints->fabric_attr->prov_name = strdup(providerName_.c_str());
        NIXL_DEBUG << "Using minimal auto-negotiated hints for " << providerName_ << " provider";
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

    ret = fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0, hints, &info);
    if (ret) {
        NIXL_ERROR << "fi_getinfo failed: " << fi_strerror(-ret);
        NIXL_DEBUG << "Trying fi_getinfo with minimal hints for provider " << providerName_;
        
        // minimal hints, see what provider supports
        struct fi_info *minimal_hints = fi_allocinfo();
        if (minimal_hints) {
            minimal_hints->fabric_attr->prov_name = strdup(providerName_.c_str());
            struct fi_info *minimal_fi = nullptr;
            int minimal_ret = fi_getinfo(FI_VERSION(1, 18), nullptr, nullptr, 0, minimal_hints, &minimal_fi);
            if (minimal_ret == 0) {
                NIXL_DEBUG << "Provider " << providerName_ << " supports: caps=0x" << std::hex << minimal_fi->caps 
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
    
    NIXL_DEBUG << "fi_ assigned successfully, checking provider info...";
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

    {
        nixl_status_t setup_status = setupEndpoint(!isConnectionless_);
        if (setup_status != NIXL_SUCCESS) {
            NIXL_ERROR << "setupEndpoint failed with status: " << setup_status;
            goto cleanup_teardown;
        }
        NIXL_DEBUG << "setupEndpoint completed successfully, isConnectionless_=" << isConnectionless_;

        // get local address
        nixl_status_t addr_status = getEndpointAddress(ep_, localAddr_);
        if (addr_status != NIXL_SUCCESS) {
            NIXL_ERROR << "getEndpointAddress() failed with status: " << addr_status;
            goto cleanup_teardown;
        }
        NIXL_DEBUG << "getEndpointAddress completed successfully, isConnectionless_=" << isConnectionless_;
    }

    // cache provider use in connect()
    cachedProviderInfo_ = fi_dupinfo(fi_);
    if (!cachedProviderInfo_) {
        NIXL_WARN << "Failed to duplicate provider info for caching";
    }

    fi_freeinfo(hints);
    // Don't free info here since fi_ = info (line 161), it will be freed in destructor

    NIXL_DEBUG << "Starting EQ event loop ...: connectionless=" << isConnectionless_;
    // start event loop thread for connection-oriented providers
    if (!isConnectionless_) {
        NIXL_DEBUG << "Creating EQ event loop thread for connection-oriented provider";
        eqThread_ = std::thread(&nixlOfiEngine::eq_event_loop, this);
    } else {
        NIXL_DEBUG << "Skipping EQ event loop thread for connectionless provider";
    }

    // init connection-focused progress thread
    connectionProgressEnabled_ = init_params->enableProgTh;
    if (init_params->pthrDelay > 0) {
        connectionProgressDelay_ = init_params->pthrDelay;
    }

    if (connectionProgressEnabled_) {
        NIXL_DEBUG << "Starting OFI connection progress thread with delay: " << connectionProgressDelay_ << " microseconds";
        connectionProgressThread_ = std::thread(&nixlOfiEngine::connectionProgressFunc, this);
    } else {
        NIXL_DEBUG << "Connection progress thread disabled, using EQ event loop only";
    }

    NIXL_DEBUG << "OFI backend constructor completed successfully";
    return;

cleanup_teardown:
    if (ep_)     { fi_close(&ep_->fid);     ep_ = nullptr; }
    if (av_)     { fi_close(&av_->fid);     av_ = nullptr; }
    if (pep_)    { fi_close(&pep_->fid);    pep_ = nullptr; }
    if (eq_)     { fi_close(&eq_->fid);     eq_ = nullptr; }
    if (cq_)     { fi_close(&cq_->fid);     cq_ = nullptr; }
    if (domain_) { fi_close(&domain_->fid); domain_ = nullptr; }
    if (fabric_) { fi_close(&fabric_->fid); fabric_ = nullptr; }
    if (fi_)     { fi_freeinfo(fi_);        fi_ = nullptr; info = nullptr; }
    if (hints)   { fi_freeinfo(hints);      hints = nullptr; }
    this->initErr = true;
}

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

// predefined provider configurations for providers that need explicit settings
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
        // Match verbs;ofi_rxm capabilities from fi_info output
        "verbs",
        FI_EP_RDM,
        FI_MSG | FI_RMA | FI_READ | FI_WRITE | FI_RECV | FI_SEND | FI_REMOTE_READ | FI_REMOTE_WRITE | FI_MULTI_RECV | FI_LOCAL_COMM | FI_REMOTE_COMM | FI_HMEM,
        0,
        FI_MR_LOCAL | FI_MR_VIRT_ADDR | FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_HMEM,
        FI_RM_ENABLED,
        {0, 0, 0, 0, 0, 0, 0, 0, FI_TC_BULK_DATA}, // tx_attr with bulk data class like fabtests
        {0, 0, 0, 0, 0, 0}, // rx_attr defaults
        FI_FORMAT_UNSPEC,
        FI_PROGRESS_AUTO,
        FI_PROGRESS_MANUAL
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
    shutdownFlag_.store(true);

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

    // stop connection progress thread
    connectionProgressStop_.store(true);
    if (connectionProgressEnabled_ && connectionProgressThread_.joinable()) {
        connectionProgressThread_.join();
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

    // note: static handles are shared across instances
    // cleanup is handled by OS when process exits
}

bool nixlOfiEngine::supportsNotif() const {
    return true;
}

bool nixlOfiEngine::supportsRemote() const {
    return true;
}

bool nixlOfiEngine::supportsLocal() const {
    return false;
}

nixl_status_t nixlOfiEngine::getNotifs(notif_list_t &notif_list) {
    if (!notif_list.empty()) {
        return NIXL_ERR_INVALID_PARAM;
    }
    
    // TODO: Implement actual OFI notification mechanism using fi_cq_read or fi_eq_read
    // For now, return empty list since OFI notifications are not yet implemented
    return NIXL_SUCCESS;
}

nixl_status_t nixlOfiEngine::genNotif(const std::string &remote_agent, const std::string &msg) const {
    // TODO: Implement actual OFI notification sending mechanism
    // This could use fi_send with a special notification message format
    // For now, return success as a no-op to satisfy the interface
    NIXL_DEBUG << "OFI genNotif stub called for agent " << remote_agent << " with message: " << msg;
    return NIXL_SUCCESS;
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

    // drive progress before attempting connection
    driveProgressIfNeeded();

    std::lock_guard<std::mutex> lock(epLock_);
    return connect_unlocked(remote_agent);
}

nixl_status_t nixlOfiEngine::connect_unlocked(const std::string &remote_agent) {
    // Note: epLock_ must already be held by caller
    NIXL_DEBUG << "connect_unlocked() called for remote_agent: " << remote_agent 
               << " isConnectionless: " << isConnectionless_;

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

    if (!domain_) {
        NIXL_ERROR << "Domain not initialized";
        delete ofi_meta;
        return NIXL_ERR_BACKEND;
    }
    
    if (mem.addr == 0 || mem.len == 0) {
        NIXL_ERROR << "Invalid memory parameters: addr=" << mem.addr << " len=" << mem.len;
        delete ofi_meta;
        return NIXL_ERR_INVALID_PARAM;
    }

    if (nixl_mem != DRAM_SEG && nixl_mem != VRAM_SEG) {
        NIXL_ERROR << "Unsupported memory type: " << nixl_mem;
        delete ofi_meta;
        return NIXL_ERR_NOT_SUPPORTED;
    }

    nixl_status_t status = NIXL_SUCCESS;
    
    // DRAM_SEG = system memory, VRAM_SEG = device memory  
    if (nixl_mem == DRAM_SEG) {
        status = registerDramMemory(mem, ofi_meta);
    } else {  // VRAM_SEG
        // determine device interface for VRAM
        uint64_t device_id = 0;
        fi_hmem_iface iface = selectHmemInterface(mem, device_id);
        
        if (iface == FI_HMEM_SYNAPSEAI) {
            status = registerSynapseAIMemoryExplicit(mem, ofi_meta);
        } else if (iface != FI_HMEM_SYSTEM) {
            status = registerHmemMemory(mem, ofi_meta, iface, device_id);
        } else {
            // VRAM requested but no device interface available - fallback
            NIXL_WARN << "VRAM requested but no HMEM interface available - falling back to system memory";
            status = registerDramMemory(mem, ofi_meta);
        }
    }

    if (status != NIXL_SUCCESS) {
        delete ofi_meta;
        return status;
    }

    if (!ofi_meta->mr) {
        NIXL_ERROR << "Memory registration returned null mr";
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
    if (!ep_) {
        NIXL_ERROR << "Primary endpoint not initialized";
        return NIXL_ERR_BACKEND;
    }
    
    fid_ep *target_ep = ep_;
    fi_addr_t dest_addr = FI_ADDR_UNSPEC;

    if (isConnectionless_) {
        auto shm_it = shmAddrs_.find(remote_agent);
        if (shm_it == shmAddrs_.end()) {
            // connection should have been established in loadRemoteConnInfo
            // if we reach here, it means the connection was not properly established
            NIXL_ERROR << "No address mapping found for " << remote_agent 
                      << " - connection should have been established in loadRemoteConnInfo";
            return NIXL_ERR_NOT_FOUND;
        }
        dest_addr = shm_it->second;
    } else {
        auto it = connectedEps_.find(remote_agent);
        if (it == connectedEps_.end()) {
            NIXL_ERROR << "OFI backend: Not connected to " << remote_agent 
                      << " - connection should have been established in loadRemoteConnInfo";
            return NIXL_ERR_NOT_FOUND;
        }
        target_ep = it->second;
        if (!target_ep) {
            NIXL_ERROR << "Connected endpoint is null for " << remote_agent;
            return NIXL_ERR_BACKEND;
        }
    }

    if (local.descCount() != remote.descCount()) {
        NIXL_ERROR << "Mismatched descriptor counts: local=" << local.descCount()
                   << ", remote=" << remote.descCount();
        return NIXL_ERR_INVALID_PARAM;
    }

    if (!cq_) {
        NIXL_ERROR << "Completion queue not initialized";
        return NIXL_ERR_BACKEND;
    }
    
    nixlOfiRequest *ofi_req = new nixlOfiRequest();
    if (!ofi_req) {
        return NIXL_ERR_BACKEND;
    }
    ofi_req->cq = cq_;
    
    if (local.descCount() <= 0) {
        NIXL_ERROR << "No descriptors to transfer";
        return NIXL_ERR_INVALID_PARAM;
    }
    
    // track posted operation contexts for proper cleanup
    std::vector<uint64_t*> op_contexts;

    for (size_t i = 0; i < static_cast<size_t>(local.descCount()); ++i) {
        const nixlMetaDesc &local_desc = local[i];
        const nixlMetaDesc &remote_desc = remote[i];

        nixlOfiMetadata *local_meta = static_cast<nixlOfiMetadata*>(local_desc.metadataP);
        nixlOfiMetadata *remote_meta = static_cast<nixlOfiMetadata*>(remote_desc.metadataP);
        
        if (!local_meta || !remote_meta || !local_meta->mr) {
            NIXL_ERROR << "Invalid metadata or memory registration";
            // clean up any previously allocated contexts
            for (auto* ctx : op_contexts) {
                delete ctx;
            }
            delete ofi_req;
            return NIXL_ERR_INVALID_PARAM;
        }
        
        // validate transfer parameters
        if (local_desc.addr == 0 || local_desc.len == 0 ||
            remote_desc.addr == 0 || remote_desc.len == 0) {
            NIXL_ERROR << "Invalid transfer parameters: local_addr=" << local_desc.addr
                      << " local_len=" << local_desc.len
                      << " remote_addr=" << remote_desc.addr 
                      << " remote_len=" << remote_desc.len;
            // clean up any previously allocated contexts
            for (auto* ctx : op_contexts) {
                delete ctx;
            }
            delete ofi_req;
            return NIXL_ERR_INVALID_PARAM;
        }
        
        if (local_desc.len != remote_desc.len) {
            NIXL_ERROR << "Length mismatch: local=" << local_desc.len 
                      << " remote=" << remote_desc.len;
            // clean up any previously allocated contexts
            for (auto* ctx : op_contexts) {
                delete ctx;
            }
            delete ofi_req;
            return NIXL_ERR_INVALID_PARAM;
        }

        // get remote memory key - either from mr or stored in desc field for remote metadata
        uint64_t remote_key;
        if (remote_meta->mr) {
            remote_key = fi_mr_key(remote_meta->mr);
        } else {
            // for remote metadata, key is stored in desc field - safe extraction
            uintptr_t desc_as_ptr = reinterpret_cast<uintptr_t>(remote_meta->desc);
            remote_key = static_cast<uint64_t>(desc_as_ptr);
        }
        
        struct fi_rma_iov rma_iov = {
            .addr = (uint64_t)remote_desc.addr,
            .len = remote_desc.len,
            .key = remote_key
        };

        // check if we can use injection for small transfers (performance optimization)
        size_t inject_size = fi_ ? fi_->tx_attr->inject_size : 0;
        bool use_inject = (local_desc.len <= inject_size) && (operation == NIXL_WRITE);

        // use unique context for each operation (not needed for inject operations)
        uint64_t* op_context = use_inject ? nullptr : new uint64_t(i);
        static uint64_t seq_num = 0;

        switch (operation) {
            case NIXL_READ:
                OFI_POST(fi_read, cq_, seq_num, "fi_read",
                        target_ep, reinterpret_cast<void*>(local_desc.addr),
                        local_desc.len, local_meta->desc, dest_addr,
                        rma_iov.addr, rma_iov.key, op_context);
                break;
            case NIXL_WRITE:
                if (use_inject) {
                    OFI_POST(fi_inject_write, cq_, seq_num, "fi_inject_write",
                            target_ep, reinterpret_cast<void*>(local_desc.addr),
                            local_desc.len, dest_addr, rma_iov.addr, rma_iov.key);
                } else {
                    OFI_POST(fi_write, cq_, seq_num, "fi_write",
                            target_ep, reinterpret_cast<void*>(local_desc.addr),
                            local_desc.len, local_meta->desc, dest_addr,
                            rma_iov.addr, rma_iov.key, op_context);
                }
                break;
            default:
                NIXL_ERROR << "Unsupported operation type";
                if (op_context) delete op_context;
                // cleanup all previously allocated contexts
                for (auto* ctx : op_contexts) {
                    delete ctx;
                }
                delete ofi_req;
                return NIXL_ERR_NOT_SUPPORTED;
        }

        // OFI_POST succeeded, track the context if not using injection
        if (op_context) {
            op_contexts.push_back(op_context);
        } else if (use_inject) {
            // injection operations complete immediately, no context tracking needed
            NIXL_DEBUG << "OFI inject transfer " << i << " completed immediately";
        }
    }

    // store context count in request for completion tracking
    ofi_req->wr_id = op_contexts.size();
    handle = ofi_req;
    return NIXL_SUCCESS;

}

void nixlOfiEngine::connectionProgressFunc() {
    NIXL_DEBUG << "OFI connection progress thread started with delay: " << connectionProgressDelay_ << " microseconds";

    while (!connectionProgressStop_.load()) {
        if (shutdownFlag_.load()) {
            break;  // shutdown in progress
        }

        // for connectionless providers (like RXM), we need to drive CQ progress
        // to handle incoming operations from RMA clients
        if (isConnectionless_ && cq_) {
            // drive CQ progress to catch incoming RMA operations
            // use multiple reads for better responsiveness with connectionless
            for (int i = 0; i < 3; i++) {
                int ret = ofi_progress_manual(cq_);
                if (ret > 0) {
                    NIXL_DEBUG << "Connection thread: processed " << ret << " completion(s)";
                } else if (ret == -FI_EAGAIN) {
                    break; // no more completions
                }
            }
        } else if (!isConnectionless_ && cq_) {
            // for connection-oriented, single CQ read is sufficient
            int ret = ofi_progress_manual(cq_);
            if (ret > 0) {
                NIXL_DEBUG << "Connection thread: processed " << ret << " completion(s)";
            }
        }

        // for connection-oriented providers, EQ events are handled by eq_event_loop
        // so this thread mainly drives CQ for incoming data

        // aggressive progress for verbs;ofi_rxm - much shorter delay
        auto delay = connectionProgressDelay_ > 0 ? connectionProgressDelay_ : 1000; // 1ms default for RXM
        std::this_thread::sleep_for(std::chrono::microseconds(delay));
    }

    NIXL_DEBUG << "OFI connection progress thread exiting";
}

void nixlOfiEngine::driveProgress() const {
    if (shutdownFlag_.load()) {
        return;  // shutdown in progress
    }

    if (!cq_) {
        return;  // no cq to drive
    }

    // fabtest-style progress driving with proper error handling
    int progress_ret = ofi_progress_manual(cq_);

    if (progress_ret > 0) {
        // successful completion read, context cleanup handled elsewhere
        NIXL_DEBUG << "Progress thread: advanced " << progress_ret << " completion(s)";
    } else if (progress_ret == -FI_EAGAIN) {
        // no completions available, normal condition
        return;
    } else if (progress_ret < 0) {
        // error occurred, rate-limit error messages
        static auto last_error_time = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_error_time).count() >= 1) {
            NIXL_DEBUG << "Progress thread error: " << fi_strerror(-progress_ret);
            last_error_time = now;
        }
    }
}

void nixlOfiEngine::driveProgressIfNeeded() const {
    auto now = std::chrono::steady_clock::now();
    auto expected = lastProgressTime_.load();

    if (now - expected >= PROGRESS_INTERVAL) {
        // Atomic compare-and-swap to prevent race condition
        if (lastProgressTime_.compare_exchange_weak(expected, now)) {
            driveProgress();
        }
    }
}

int nixlOfiEngine::ofi_progress_manual(fid_cq *cq) const {
    if (!cq) {
        return -FI_EINVAL;
    }

    // read multiple completions for better RXM performance
    const size_t batch_size = 8;
    struct fi_cq_data_entry comps[batch_size];
    int total_processed = 0;

    // try to read multiple completions in batches
    for (int batch = 0; batch < 3; batch++) {
        int ret = fi_cq_read(cq, comps, batch_size);

        if (ret > 0) {
            total_processed += ret;
            continue; // try to read more
        } else if (ret == -FI_EAGAIN) {
            break; // no more completions
        } else if (ret == -FI_EAVAIL) {
            struct fi_cq_err_entry err_entry;
            ret = fi_cq_readerr(cq, &err_entry, 0);
            if (ret < 0) {
                if (!shutdownFlag_.load()) {
                    NIXL_ERROR << "fi_cq_readerr failed: " << fi_strerror(-ret);
                }
            } else {
                // Don't log cancellation errors during shutdown as errors
                if (err_entry.err == FI_ECANCELED && shutdownFlag_.load()) {
                    NIXL_DEBUG << "Operation canceled during shutdown: " << fi_strerror(err_entry.err);
                } else if (err_entry.err == FI_EIO && shutdownFlag_.load()) {
                    NIXL_DEBUG << "I/O error during shutdown: " << fi_strerror(err_entry.err);
                } else {
                    NIXL_ERROR << "CQ error: " << fi_strerror(err_entry.err);
                }
            }
            return -FI_EAVAIL;
        } else {
            if (!shutdownFlag_.load()) {
                NIXL_ERROR << "fi_cq_read failed: " << fi_strerror(-ret);
            }
            return ret;
        }
    }

    return total_processed > 0 ? total_processed : -FI_EAGAIN;
}

nixl_status_t nixlOfiEngine::checkXfer(nixlBackendReqH* handle) const {
    if (shutdownFlag_.load()) {
        return NIXL_ERR_BACKEND;  // shutdown in progress
    }

    // rate-limited progress driving for main thread
    driveProgressIfNeeded();

    nixlOfiRequest *ofi_req = static_cast<nixlOfiRequest*>(handle);
    if (!ofi_req || !ofi_req->cq) {
        return NIXL_ERR_INVALID_PARAM;
    }

    uint64_t expected_completions = ofi_req->wr_id.load();
    if (expected_completions == 0) {
        return NIXL_SUCCESS; // no operations were posted
    }

    // read available completions in batches
    const size_t batch_size = 16;
    struct fi_cq_data_entry entries[batch_size];
    size_t max_read = std::min(expected_completions, batch_size);
    int ret = fi_cq_read(ofi_req->cq, entries, max_read);
    
    if (ret > 0) {
        NIXL_DEBUG << "checkXfer: got " << ret << " completions";
        // got some completions - free the contexts
        for (int i = 0; i < ret; ++i) {
            if (entries[i].op_context) {
                delete static_cast<uint64_t*>(entries[i].op_context);
            }
        }
        
        // thread-safe atomic update of remaining completions
        uint64_t expected = ofi_req->wr_id.load();
        uint64_t new_count;
        
        do {
            if (expected >= static_cast<uint64_t>(ret)) {
                new_count = expected - ret;
            } else {
                NIXL_ERROR << "Completion count underflow: expected=" << expected << " got=" << ret;
                new_count = 0;
            }
        } while (!ofi_req->wr_id.compare_exchange_weak(expected, new_count));
        
        if (ofi_req->wr_id.load() == 0) {
            return NIXL_SUCCESS; // all operations completed
        }
        return NIXL_IN_PROG; // some operations still pending
    } else if (ret == -FI_EAGAIN) {
        return NIXL_IN_PROG;
    } else if (ret == -FI_EAVAIL) {
        // handle error completions like fabtest ft_progress pattern
        struct fi_cq_err_entry err_entry;
        int err_ret = fi_cq_readerr(ofi_req->cq, &err_entry, 0);
        if (err_ret > 0) {
            // Don't log cancellation errors during shutdown as errors
            if (err_entry.err == FI_ECANCELED && shutdownFlag_.load()) {
                NIXL_DEBUG << "checkXfer: Operation canceled during shutdown";
            } else if (err_entry.err == FI_EIO && shutdownFlag_.load()) {
                NIXL_DEBUG << "checkXfer: I/O error during shutdown";
            } else {
                NIXL_ERROR << "CQ error completion: " << fi_strerror(err_entry.err) << " provider=" << err_entry.prov_errno;
            }

            // cleanup context on error and count as completion
            if (err_entry.op_context) {
                delete static_cast<uint64_t*>(err_entry.op_context);
            }

            // atomically decrement remaining operations
            uint64_t expected = ofi_req->wr_id.load();
            uint64_t new_count = expected > 0 ? expected - 1 : 0;
            while (!ofi_req->wr_id.compare_exchange_weak(expected, new_count)) {
                new_count = expected > 0 ? expected - 1 : 0;
            }
        } else {
            NIXL_ERROR << "fi_cq_readerr failed: " << fi_strerror(-err_ret);
        }
        return NIXL_ERR_BACKEND;
    } else if (ret < 0) {
        NIXL_ERROR << "fi_cq_read failed: " << fi_strerror(-ret);
        return NIXL_ERR_BACKEND;
    }
    return NIXL_IN_PROG;
}

nixl_status_t nixlOfiEngine::releaseReqH(nixlBackendReqH* handle) const {
    nixlOfiRequest *ofi_req = static_cast<nixlOfiRequest*>(handle);
    if (!ofi_req) {
        return NIXL_ERR_INVALID_PARAM;
    }

    // try to drain pending completions to prevent context leaks
    uint64_t pending_ops = ofi_req->wr_id.load();
    if (pending_ops > 0) {
        int drain_attempts = 0;
        // dynamic max attempts based on pending operations, with reasonable bounds
        const int base_attempts = 50;
        const int max_attempts = std::min(static_cast<int>(pending_ops * 5), 500);
        const int total_attempts = std::max(base_attempts, max_attempts);

        NIXL_DEBUG << "Draining " << pending_ops << " pending operations (max attempts: " << total_attempts << ")";

        while (ofi_req->wr_id.load() > 0 && drain_attempts < total_attempts) {
            // drain completions in batches
            const size_t batch_size = 16;
            struct fi_cq_data_entry entries[batch_size];
            uint64_t expected_completions = ofi_req->wr_id.load();
            size_t max_read = std::min(expected_completions, batch_size);

            int ret = fi_cq_read(ofi_req->cq, entries, max_read);
            if (ret > 0) {
                // process completions and free contexts
                // for (int i = 0; i < ret; ++i) {
                //     if (entries[i].op_context) {
                //         delete static_cast<uint64_t*>(entries[i].op_context);
                //     }
                // }

                // atomic update of completion count using compare-exchange to prevent races
                uint64_t expected = ofi_req->wr_id.load();
                uint64_t new_count;
                do {
                    new_count = (expected >= static_cast<uint64_t>(ret)) ? expected - ret : 0;
                } while (!ofi_req->wr_id.compare_exchange_weak(expected, new_count));

                NIXL_DEBUG << "Drained " << ret << " completions, " << new_count << " remaining";
            } else if (ret == -FI_EAGAIN) {
                // no completions available, wait briefly and retry
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            } else if (ret == -FI_EAVAIL) {
                // handle error completions
                struct fi_cq_err_entry err_entry;
                int err_ret = fi_cq_readerr(ofi_req->cq, &err_entry, 0);
                if (err_ret > 0) {
                    if (shutdownFlag_.load()) {
                        NIXL_DEBUG << "Error completion during shutdown: " << fi_strerror(err_entry.err);
                    } else {
                        NIXL_WARN << "Error completion in releaseReqH: " << fi_strerror(err_entry.err);
                    }

                    // cleanup context on error and count as completion
                    if (err_entry.op_context) {
                        delete static_cast<uint64_t*>(err_entry.op_context);
                    }

                    // atomically decrement remaining operations
                    uint64_t expected = ofi_req->wr_id.load();
                    uint64_t new_count;
                    do {
                        new_count = expected > 0 ? expected - 1 : 0;
                    } while (!ofi_req->wr_id.compare_exchange_weak(expected, new_count));
                }
            } else {
                // other CQ errors - log but continue trying for a few more attempts
                if (!shutdownFlag_.load()) {
                    NIXL_WARN << "CQ read error in releaseReqH: " << fi_strerror(-ret);
                }
                // give a few more chances for transient errors
                if (drain_attempts < total_attempts - 10) {
                    std::this_thread::sleep_for(std::chrono::microseconds(500));
                } else {
                    break; // stop if we're near the limit
                }
            }

            drain_attempts++;
        }

        // log final status
        uint64_t remaining = ofi_req->wr_id.load();
        if (remaining > 0) {
            if (shutdownFlag_.load()) {
                NIXL_DEBUG << "Shutdown: releasing request with " << remaining
                          << " pending operations after " << drain_attempts << " attempts";
            } else {
                NIXL_WARN << "Releasing request with " << remaining
                         << " pending operations after " << drain_attempts
                         << " attempts (potential context leak)";
            }
        } else {
            NIXL_DEBUG << "Successfully drained all operations after " << drain_attempts << " attempts";
        }
    }

    delete ofi_req;
    return NIXL_SUCCESS;
}

nixl_status_t nixlOfiEngine::getConnInfo(std::string &conn_info) const {
    // drive progress before returning connection info
    driveProgressIfNeeded();

    conn_info = localAddr_;
    return NIXL_SUCCESS;
}

nixl_status_t nixlOfiEngine::loadRemoteConnInfo(const std::string &remote_agent, const std::string &conn_info) {
    // drive progress to handle any pending operations before loading new connection info
    driveProgressIfNeeded();

    // validate remote agent name to prevent memory attacks
    if (remote_agent.empty() || remote_agent.size() > 256) {
        NIXL_ERROR << "Invalid remote agent name length: " << remote_agent.size();
        return NIXL_ERR_INVALID_PARAM;
    }

    if (conn_info.empty() || conn_info.size() > 1024) {
        NIXL_ERROR << "Invalid connection info size: " << conn_info.size();
        return NIXL_ERR_INVALID_PARAM;
    }
    
    std::lock_guard<std::mutex> lock(epLock_);
    remoteAddrs_[remote_agent] = conn_info;
    
    // establish connection immediately when remote agent info is loaded
    NIXL_DEBUG << "Establishing connection to " << remote_agent << " immediately";
    nixl_status_t connect_status = connect_unlocked(remote_agent);
    if (connect_status != NIXL_SUCCESS) {
        NIXL_ERROR << "Failed to establish connection to " << remote_agent << " during loadRemoteConnInfo";
        // Remove the address entry since connection failed
        remoteAddrs_.erase(remote_agent);
        return connect_status;
    }
    
    NIXL_DEBUG << "Successfully established connection to " << remote_agent;
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
    
    // validate input metadata
    if (input.metaInfo.empty() || input.metaInfo.size() > 32) {
        delete remote_meta;
        NIXL_ERROR << "Invalid metadata size: " << input.metaInfo.size();
        return NIXL_ERR_INVALID_PARAM;
    }
    
    // parse the memory key from the metadata string
    try {
        // copy metaInfo to ensure null termination and lifetime safety
        std::string key_str(input.metaInfo.begin(), input.metaInfo.end());
        uint64_t remote_key = std::stoull(key_str);
        
        // validate remote key before storing
        if (remote_key == 0 || remote_key == UINT64_MAX) {
            delete remote_meta;
            NIXL_ERROR << "Invalid remote memory key: " << remote_key;
            return NIXL_ERR_INVALID_PARAM;
        }
        
        // for remote metadata, we don't have an actual mr object, just the key
        // store the key for later use in RMA operations - safe conversion
        remote_meta->mr = nullptr;  // No local mr for remote metadata
        remote_meta->desc = reinterpret_cast<void*>(static_cast<uintptr_t>(remote_key));
        
        output = remote_meta;
        return NIXL_SUCCESS;
    } catch (...) {
        delete remote_meta;
        NIXL_ERROR << "Failed to parse remote memory key";
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
    NIXL_DEBUG << "isConnectionlessProvider: checking fi_=" << (fi_ ? "valid" : "null") 
               << " ep_attr=" << (fi_ && fi_->ep_attr ? "valid" : "null");
    
    if (fi_ && fi_->ep_attr) {
        enum fi_ep_type ep_type = fi_->ep_attr->type;
        bool is_connectionless = (ep_type == FI_EP_RDM || ep_type == FI_EP_DGRAM);
        NIXL_DEBUG << "isConnectionlessProvider: ep_type=" << ep_type 
                   << " (RDM=" << FI_EP_RDM << " DGRAM=" << FI_EP_DGRAM << ")"
                   << " result=" << is_connectionless;
        return is_connectionless;
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
        NIXL_DEBUG << "TCP passive endpoint listening for connections";
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
        NIXL_DEBUG << "Provider " << provider_name << " does not support generic HMEM";
        
        // Special case: verbs can support SynapseAI through DMA buffers
        // even without advertising FI_HMEM capability.
        // Evidence: vrb_read_params() logs "dmabuf support is enabled" for verbs
        if (provider_name == "verbs") {
            synapseai_supported = true;
        } else {
            synapseai_supported = false;
        }
        
        cuda_supported = false;
        ze_supported = false;
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
    
    // helper for safe environment variable checking
    auto isEnvTrue = [](const char* env_val) -> bool {
        return env_val && (strcmp(env_val, "1") == 0 || strcmp(env_val, "true") == 0);
    };
    
    // Synapse device validation
    auto validateSynapseAIDevice = [](uint64_t dev_id) -> bool {
        std::string device_path = "/dev/accel/accel" + std::to_string(dev_id);
        if (access(device_path.c_str(), R_OK | W_OK) != 0) {
            NIXL_INFO << "SynapseAI device " << device_path << " not accessible, will fallback to system memory";
            return false;
        }
        return true;
    };
    
    auto validateCudaDevice = [](uint64_t dev_id) -> bool {
        // TODO: add proper CUDA device validation
        return true;
    };
    
    auto validateZeDevice = [](uint64_t dev_id) -> bool {
        // TODO: add proper ZE device validation
        return true;
    };
    
    // HMEM interface configuration
    struct HmemConfig {
        fi_hmem_iface iface;
        bool supported;
        const char* name;
        const char* explicit_env;      // HMEM_SYNAPSEAI, HMEM_CUDA, etc.
        const char* implicit_env;      // HABANA_VISIBLE_DEVICES, CUDA_VISIBLE_DEVICES, etc.
        std::function<bool(uint64_t)> validate;
    };
    
    const HmemConfig configs[] = {
        {
            FI_HMEM_SYNAPSEAI, 
            hmemSynapseaiSupported_, 
            "SynapseAI",
            "HMEM_SYNAPSEAI", 
            "HABANA_VISIBLE_DEVICES",
            validateSynapseAIDevice
        },
        {
            FI_HMEM_CUDA, 
            hmemCudaSupported_, 
            "CUDA",
            "HMEM_CUDA", 
            "CUDA_VISIBLE_DEVICES",
            validateCudaDevice
        },
        {
            FI_HMEM_ZE, 
            hmemZeSupported_, 
            "ZE",
            "HMEM_ZE", 
            "ZE_AFFINITY_MASK",
            validateZeDevice
        }
    };
    
    // 1 check explicit environment variable overrides first
    for (const auto& config : configs) {
        const char* explicit_env = getenv(config.explicit_env);
        if (isEnvTrue(explicit_env)) {
            if (!config.supported) {
                NIXL_ERROR << config.explicit_env << " set but " << config.name 
                          << " interface not supported by provider";
                return FI_HMEM_SYSTEM;
            }
            if (!config.validate(device_id)) {
                return FI_HMEM_SYSTEM;
            }
            NIXL_INFO << "Using " << config.name << " HMEM interface for device " 
                     << device_id << " (via " << config.explicit_env << ")";
            return config.iface;
        }
    }
    
    // 2 check implicit environment variables
    for (const auto& config : configs) {
        const char* implicit_env = getenv(config.implicit_env);
        if (implicit_env && config.supported) {
            if (!config.validate(device_id)) {
                if (config.iface == FI_HMEM_SYNAPSEAI) {
                    NIXL_ERROR << "  unset " << config.implicit_env << " to use host memory";
                }
                return FI_HMEM_SYSTEM;
            }
            NIXL_INFO << "Using " << config.name << " HMEM interface for device " 
                     << device_id << " (via " << config.implicit_env << ")";
            return config.iface;
        }
    }
    
    // 3 auto-select from supported interfaces
    NIXL_INFO << "No HMEM environment variables detected, auto-selecting interface for VRAM";
    for (const auto& config : configs) {
        if (config.supported && config.validate(device_id)) {
            NIXL_INFO << "Auto-selected " << config.name << " HMEM interface for device " << device_id;
            return config.iface;
        }
    }
    
    NIXL_WARN << "No HMEM interfaces supported. Falling back to host memory registration";
    return FI_HMEM_SYSTEM;
}

nixl_status_t nixlOfiEngine::registerDramMemory(const nixlBlobDesc &mem, nixlOfiMetadata *ofi_meta) const {
    uint64_t access_flags = getMemoryRegistrationAccessFlags(fi_);
    
    int ret = fi_mr_reg(domain_, reinterpret_cast<void*>(mem.addr), mem.len,
                       access_flags, 0, 0, 0, &ofi_meta->mr, nullptr);
    
    if (ret) {
        NIXL_ERROR << "fi_mr_reg failed for system memory: " << fi_strerror(-ret);
        ofi_meta->mr = nullptr;
        return NIXL_ERR_BACKEND;
    }
    
    return NIXL_SUCCESS;
}

nixl_status_t nixlOfiEngine::registerHmemMemory(const nixlBlobDesc &mem, nixlOfiMetadata *ofi_meta, fi_hmem_iface iface, uint64_t device_id) const {
    if (device_id >= UINT32_MAX) {
        NIXL_ERROR << "Invalid device ID: " << device_id;
        return NIXL_ERR_INVALID_PARAM;
    }
    
    struct fi_mr_attr mr_attr = {};
    struct iovec iov = {};
    
    iov.iov_base = reinterpret_cast<void*>(mem.addr);
    iov.iov_len = mem.len;
    
    mr_attr.mr_iov = &iov;
    mr_attr.iov_count = 1;
    mr_attr.access = getMemoryRegistrationAccessFlags(fi_);
    mr_attr.iface = iface;
    
    switch (iface) {
        case FI_HMEM_CUDA:
            mr_attr.device.cuda = static_cast<uint32_t>(device_id);
            break;
        case FI_HMEM_ZE:
            mr_attr.device.ze = static_cast<uint32_t>(device_id);
            break;
        case FI_HMEM_SYNAPSEAI:
            mr_attr.device.synapseai = static_cast<uint32_t>(device_id);
            break;
        default:
            NIXL_ERROR << "Unsupported HMEM interface: " << iface;
            return NIXL_ERR_NOT_SUPPORTED;
    }
    
    int ret = fi_mr_regattr(domain_, &mr_attr, 0, &ofi_meta->mr);
    if (ret) {
        NIXL_ERROR << "fi_mr_regattr failed for HMEM: " << fi_strerror(-ret);
        ofi_meta->mr = nullptr;
        return NIXL_ERR_BACKEND;
    }
    
    return NIXL_SUCCESS;
}


nixl_status_t nixlOfiEngine::registerSynapseAIMemoryExplicit(const nixlBlobDesc &mem, nixlOfiMetadata *ofi_meta) const {
    // Try to get device info from the memory descriptor first
    // If mem.devId is a valid SynapseAI device handle, use it directly
    synDeviceId device_id = static_cast<synDeviceId>(mem.devId);
    synDeviceInfoV2 device_info;
    
    // Try to get device info directly using the device ID from memory descriptor
    NIXL_DEBUG << "Attempting to get device info for device ID: " << device_id;
    
    // thread-safe initialization of static handles
    std::lock_guard<std::mutex> lock(synapseai_init_mutex_);
    
    // load synapseAI library functions (shared across instances)
    if (!synapseai_handle_) {
        synapseai_handle_ = dlopen("libSynapse.so", RTLD_NOW);
        if (!synapseai_handle_) {
            NIXL_ERROR << "failed to dlopen libSynapse.so: " << dlerror();
            return NIXL_ERR_BACKEND;
        }
        
        synapseai_ops_.synDeviceGetInfoV2 = 
            (synStatus (*)(const synDeviceId, synDeviceInfoV2 *))dlsym(synapseai_handle_, "synDeviceGetInfoV2");
        if (!synapseai_ops_.synDeviceGetInfoV2) {
            NIXL_ERROR << "failed to find synDeviceGetInfoV2: " << dlerror();
            return NIXL_ERR_BACKEND;
        }
    }
    
    if (!hlthunk_handle_) {
        hlthunk_handle_ = dlopen("libhl-thunk.so", RTLD_NOW);
        if (!hlthunk_handle_) {
            NIXL_ERROR << "failed to dlopen libhl-thunk.so: " << dlerror();
            return NIXL_ERR_BACKEND;
        }
        
        synapseai_ops_.hlthunk_device_mapped_memory_export_dmabuf_fd = 
            (int (*)(int, uint64_t, uint64_t, uint64_t, uint32_t))dlsym(hlthunk_handle_, "hlthunk_device_mapped_memory_export_dmabuf_fd");
        if (!synapseai_ops_.hlthunk_device_mapped_memory_export_dmabuf_fd) {
            NIXL_ERROR << "failed to find hlthunk_device_mapped_memory_export_dmabuf_fd: " << dlerror();
            return NIXL_ERR_BACKEND;
        }
    }
    
    // Check if device is available first
    if (synapseai_ops_.synDeviceGetInfoV2(device_id, &device_info) != synSuccess) {
        NIXL_INFO << "SynapseAI device " << device_id << " not available, falling back to DRAM registration";
        return registerDramMemory(mem, ofi_meta);
    }
    
    NIXL_INFO << "Using existing SynapseAI device (PyTorch initialized) ID: " << device_id;

    // Calculate aligned buffer size
    const size_t ACCEL_PAGE_SIZE = 4096;
    size_t modi_memlen = mem.len;
    
    // Check if memory is within device range
    uint64_t hbm_base = device_info.globalHbmBaseAddress;
    uint64_t hbm_size = device_info.dramSize;
    
    NIXL_DEBUG << "Memory validation: addr=0x" << std::hex << mem.addr 
              << " HBM_base=0x" << hbm_base 
              << " HBM_size=0x" << hbm_size << std::dec;
    
    if (mem.addr < hbm_base || mem.addr >= (hbm_base + hbm_size)) {
        NIXL_ERROR << "Memory address 0x" << std::hex << mem.addr 
                  << " is not within HPU device memory range [0x" << hbm_base 
                  << " - 0x" << (hbm_base + hbm_size) << "]";
        return NIXL_ERR_INVALID_PARAM;
    }
    
    // Align device offset to suit page size
    uint64_t device_offset = mem.addr - hbm_base;
    uint64_t modi_mem_addr;
    if (mem.addr % ACCEL_PAGE_SIZE) {
        modi_mem_addr = (mem.addr / ACCEL_PAGE_SIZE) * ACCEL_PAGE_SIZE;
        device_offset -= mem.addr - modi_mem_addr;
        modi_memlen += ACCEL_PAGE_SIZE;
    }
    modi_memlen = (modi_memlen + ACCEL_PAGE_SIZE - 1) & ~(ACCEL_PAGE_SIZE - 1);

    NIXL_INFO << "Exporting dmabuf: fd=" << device_info.fd
              << " base=0x" << std::hex << hbm_base 
              << " size=" << std::dec << modi_memlen
              << " tensor data ptr=0x" << std::hex << mem.addr
              << " modified tensor data ptr=0x" << std::hex << modi_mem_addr
              << " offset=0x" << std::hex << device_offset;

    // Get dmabuf fd
    int dmabuf_fd = synapseai_ops_.hlthunk_device_mapped_memory_export_dmabuf_fd(
        device_info.fd,
        hbm_base,
        modi_memlen,
        device_offset,
        (O_RDWR | O_CLOEXEC)
    );
    
    if (dmabuf_fd < 0) {
        NIXL_ERROR << "hlthunk_device_mapped_memory_export_dmabuf_fd failed: " << strerror(-dmabuf_fd);
        NIXL_ERROR << "  device_fd=" << device_info.fd;
        NIXL_ERROR << "  base_addr=0x" << std::hex << hbm_base;
        NIXL_ERROR << "  size=" << std::dec << modi_memlen;
        NIXL_ERROR << "  offset=0x" << std::hex << device_offset;
        return NIXL_ERR_BACKEND;
    }
    
    NIXL_DEBUG << "Got dmabuf_fd: " << dmabuf_fd << " for device memory addr: 0x" 
              << std::hex << mem.addr << " size: " << std::dec << mem.len;
    
    // set up dmabuf structure - fix page alignment issue
    // kernel exported page-aligned region, but we register exact buffer
    struct fi_mr_dmabuf dmabuf = {};
    dmabuf.fd = dmabuf_fd;
    dmabuf.offset = 0;                                   // kernel handled offset
    dmabuf.len = modi_memlen;                                // exact buffer size
    dmabuf.base_addr = reinterpret_cast<void*>(modi_mem_addr); // exact buffer start
    
    // Set up memory registration attributes
    struct fi_mr_attr mr_attr = {};
    mr_attr.dmabuf = &dmabuf;
    mr_attr.iov_count = 1;
    mr_attr.access = getMemoryRegistrationAccessFlags(fi_);
    mr_attr.iface = FI_HMEM_SYNAPSEAI;
    mr_attr.device.synapseai = static_cast<uint32_t>(device_id);
    
    NIXL_DEBUG << "Registering SynapseAI memory with explicit dmabuf fd: " << dmabuf_fd;
    
    // register memory with explicit dmabuf
    int ret = fi_mr_regattr(domain_, &mr_attr, FI_MR_DMABUF, &ofi_meta->mr);
    
    // cleanup fd after registration
    close(dmabuf_fd);
    
    if (ret) {
        NIXL_ERROR << "memory registration failed: " << fi_strerror(-ret);
        ofi_meta->mr = nullptr;
        return NIXL_ERR_BACKEND;
    }
    
    // set descriptor
    ofi_meta->desc = fi_mr_desc(ofi_meta->mr);
    if (!ofi_meta->desc) {
        NIXL_ERROR << "fi_mr_desc failed";
        fi_close(&ofi_meta->mr->fid);
        ofi_meta->mr = nullptr;
        return NIXL_ERR_BACKEND;
    }
    
    NIXL_INFO << "successfully registered SynapseAI memory via dmabuf";
    return NIXL_SUCCESS;
}


