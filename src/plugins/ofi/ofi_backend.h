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

#ifndef __OFI_BACKEND_H
#define __OFI_BACKEND_H

#include <nixl.h>
#include <nixl_types.h>
#include "backend/backend_engine.h"

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_eq.h>
// #include <rdma/fi_cq.h>
#include <rdma/fi_ext.h>

#include <string>
#include <map>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>

class nixlOfiMetadata : public nixlBackendMD {
public:
    fid_mr *mr;
    void *desc;

    nixlOfiMetadata() : nixlBackendMD(false), mr(nullptr), desc(nullptr) { }
    ~nixlOfiMetadata() { }
};

class nixlOfiRequest : public nixlBackendReqH {
public:
    fid_cq *cq;
    uint64_t wr_id;

    nixlOfiRequest() : cq(nullptr), wr_id(0) { }
    ~nixlOfiRequest() { }
};

class nixlOfiEngine : public nixlBackendEngine {
public:
    // constructors and destructor
    nixlOfiEngine(const nixlBackendInitParams* init_params);
    ~nixlOfiEngine();

    // member functions
    bool supportsNotif() const override;
    bool supportsRemote() const override;
    bool supportsLocal() const override;
    bool supportsProgTh() const override;

    nixl_mem_list_t getSupportedMems() const override;

    nixl_status_t connect(const std::string &remote_agent) override;
    nixl_status_t disconnect(const std::string &remote_agent) override;

    nixl_status_t registerMem(const nixlBlobDesc &mem,
                             const nixl_mem_t &nixl_mem,
                             nixlBackendMD* &out) override;
    nixl_status_t deregisterMem(nixlBackendMD *meta) override;
    nixl_status_t unloadMD(nixlBackendMD* input) override;

    nixl_status_t prepXfer(const nixl_xfer_op_t &operation,
                          const nixl_meta_dlist_t &local,
                          const nixl_meta_dlist_t &remote,
                          const std::string &remote_agent,
                          nixlBackendReqH* &handle,
                          const nixl_opt_b_args_t* opt_args=nullptr) const override;
    nixl_status_t postXfer(const nixl_xfer_op_t &operation,
                          const nixl_meta_dlist_t &local,
                          const nixl_meta_dlist_t &remote,
                          const std::string &remote_agent,
                          nixlBackendReqH* &handle,
                          const nixl_opt_b_args_t* opt_args=nullptr) const override;

    nixl_status_t checkXfer(nixlBackendReqH* handle) const override;
    nixl_status_t releaseReqH(nixlBackendReqH* handle) const override;

    nixl_status_t getConnInfo(std::string &conn_info) const override;
    nixl_status_t loadRemoteConnInfo(const std::string &remote_agent, const std::string &conn_info) override;
    
    nixl_status_t getPublicData(const nixlBackendMD* meta, std::string &str) const override;
    nixl_status_t loadRemoteMD(const nixlBlobDesc &input, const nixl_mem_t &nixl_mem, 
                               const std::string &remote_agent, nixlBackendMD* &output) override;

private:
    // type definitions and nested classes
    struct ProviderConfig {
        std::string name;
        enum fi_ep_type ep_type;
        uint64_t caps;
        uint64_t mode;
        uint64_t mr_mode;
        fi_resource_mgmt resource_mgmt;
        struct fi_tx_attr tx_attr;
        struct fi_rx_attr rx_attr;
        uint32_t addr_format;
        enum fi_progress data_progress;
        enum fi_progress control_progress;
    };
    
    // static member variables
    static const ProviderConfig SUPPORTED_PROVIDERS[];
    static const size_t NUM_SUPPORTED_PROVIDERS;

    // member functions
    void eq_event_loop();
    bool isConnectionlessProvider() const;
    nixl_status_t setupEndpoint(bool connection_oriented);
    static nixl_status_t getEndpointAddress(fid_ep* endpoint, std::string& address);
    static void detectHmemCapabilities(struct fi_info* fi_info,
                                       const std::string& provider_name,
                                       bool& cuda_supported,
                                       bool& ze_supported,
                                       bool& synapseai_supported);
    static const ProviderConfig* findProviderConfig(const std::string& provider_name);
    
    // parameter helpers
    void getStringParam(const nixlBackendInitParams* init_params, const std::string& key, std::string& value);
    void getLongParam(const nixlBackendInitParams* init_params, const std::string& key, long& value, long min_val, long max_val);
    void getSizeTParam(const nixlBackendInitParams* init_params, const std::string& key, size_t& value);
    
    void configureHintsForProvider(struct fi_info* hints, const std::string& provider_name);

    // data members
    fid_fabric *fabric_;
    fid_domain *domain_;
    fid_ep *ep_;
    fid_cq *cq_;
    fid_eq *eq_;
    fid_pep *pep_;
    struct fi_info *fi_;

    std::string providerName_;
    struct fi_info *cachedProviderInfo_;
    std::string localAddr_;
    std::map<std::string, std::string> remoteAddrs_;
    std::map<std::string, fid_ep *> connectedEps_;
    std::map<std::string, fi_addr_t> shmAddrs_;
    fid_av *av_;
    mutable std::mutex epLock_;
    bool isConnectionless_;

    std::thread eqThread_;
    std::atomic<bool> eqThreadStop_;
    std::atomic<bool> eqThreadPaused_;
    std::mutex eqPauseMutex_;
    std::condition_variable eqPauseCV_;
    long eqTimeoutMs_;
    bool hmemZeSupported_;
    bool hmemCudaSupported_;
    bool hmemSynapseaiSupported_;

    std::string localAgentName_;
};

#endif
