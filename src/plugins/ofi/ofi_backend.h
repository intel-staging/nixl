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

class nixlOFI_Metadata : public nixlBackendMD {
public:
    fid_mr *mr;
    void *desc;

    nixlOFI_Metadata() : nixlBackendMD(false), mr(nullptr), desc(nullptr) { }
    ~nixlOFI_Metadata() { }
};

class nixlOFI_Request : public nixlBackendReqH {
public:
    fid_cq *cq;
    uint64_t wr_id;

    nixlOFI_Request() : cq(nullptr), wr_id(0) { }
    ~nixlOFI_Request() { }
};

class nixlOFI_Engine : public nixlBackendEngine {
private:
    fid_fabric *fabric;
    fid_domain *domain;
    fid_ep *ep;
    fid_cq *cq;
    fid_eq *eq;
    fid_pep *pep;
    struct fi_info *fi;

    std::string provider_name;
    struct fi_info *cached_provider_info;
    std::string local_addr;
    std::map<std::string, std::string> remote_addrs;
    std::map<std::string, fid_ep *> connected_eps;
    std::map<std::string, fi_addr_t> shm_addrs;
    fid_av *av;
    mutable std::mutex ep_lock;
    bool is_connectionless;

    std::thread eq_thread;
    std::atomic<bool> eq_thread_stop;
    long eq_timeout_ms;
    bool hmem_ze_supported;
    bool hmem_cuda_supported;
    bool hmem_synapseai_supported;

    std::string local_agent_name;

    void eq_event_loop();
    bool isConnectionlessProvider() const;
    nixl_status_t setupEndpoint(bool connection_oriented);
    static nixl_status_t getEndpointAddress(fid_ep* endpoint, std::string& address);
    static void detectHmemCapabilities(struct fi_info* fi_info,
                                       const std::string& provider_name,
                                       bool& cuda_supported,
                                       bool& ze_supported,
                                       bool& synapseai_supported);
    
    // provider config based on fabtests test_configs
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
        uint32_t data_progress;
        uint32_t control_progress;
    };
    
    static const ProviderConfig SUPPORTED_PROVIDERS[];
    static const size_t NUM_SUPPORTED_PROVIDERS;
    static const ProviderConfig* findProviderConfig(const std::string& provider_name);
    
    // parameter helpers
    void getStringParam(const nixlBackendInitParams* init_params, const std::string& key, std::string& value);
    void getLongParam(const nixlBackendInitParams* init_params, const std::string& key, long& value, long min_val, long max_val);
    void getSizeTParam(const nixlBackendInitParams* init_params, const std::string& key, size_t& value);
    
    void configureHintsForProvider(struct fi_info* hints, const std::string& provider_name);

public:
    nixlOFI_Engine(const nixlBackendInitParams* init_params);
    ~nixlOFI_Engine();

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
};

#endif
