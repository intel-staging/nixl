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

nixlOFI_Engine::nixlOFI_Engine(const nixlBackendInitParams* init_params) :
    fabric(nullptr),
    domain(nullptr),
    ep(nullptr),
    cq(nullptr),
    eq(nullptr),
    fi(nullptr),
    cached_provider_info(nullptr),
    eq_thread_stop(false),
    eq_timeout_ms(100),
{
    local_agent_name = init_params->localAgentName;
    struct fi_info *hints = nullptr;
    struct fi_info *info = nullptr;
    int ret = 0;

    // provider name, default: "verbs;ofi_rxm"
    auto it = init_params->customParams->find("provider");
    if (it != init_params->customParams->end()) {
        provider_name = it->second;
    }

    it = init_params->customParams->find("eq_timeout_ms");
    if (it != init_params->customParams->end()) {
        try {
            long timeout = std::stol(it->second);
            if (timeout >= 0 && timeout <= 60000) {  // 0-60 seconds max
                eq_timeout_ms = timeout;
            } else {
                NIXL_WARN << "eq_timeout_ms out of range [0-60000]: " << timeout << ", using default 100ms";
            }
        } catch (const std::exception& e) {
            NIXL_WARN << "Invalid eq_timeout_ms parameter: " << it->second << ", using default 100ms";
        }
    }

    hints = fi_allocinfo();
    if (!hints) {
        setInitErr(NIXL_ERR_BACKEND);
        NIXL_ERROR << "fi_allocinfo failed";
        return;
    }

    hints->caps = FI_MSG | FI_RMA | FI_READ | FI_WRITE | FI_DIRECTED_CM;
    hints->mode = FI_CONTEXT | FI_RX_CQ_DATA;
    hints->ep_attr->type = FI_EP_RDM; // Reliable Datagram Messaging

    ret = fi_getinfo(FI_VERSION(1, 0), nullptr, nullptr, 0, hints, &info);
    if (ret) {
        NIXL_ERROR << "fi_getinfo failed: " << fi_strerror(-ret);
        goto err_getinfo;
    }

    // find the desired provider
    struct fi_info *current_info = info;
    while (current_info) {
        if (current_info->fabric_attr->prov_name &&
            provider_name == current_info->fabric_attr->prov_name) {
            fi = current_info;
            break;
        }
        current_info = current_info->next;
    }

    if (!fi) {
        NIXL_ERROR << "Provider " << provider_name << " not found";
        goto err_getinfo;
    }


    ret = fi_fabric(fi->fabric_attr, &fabric, nullptr);
    if (ret) {
        NIXL_ERROR << "fi_fabric failed: " << fi_strerror(-ret);
        goto err_getinfo;
    }

    ret = fi_domain(fabric, fi, &domain, nullptr);
    if (ret) {
        NIXL_ERROR << "fi_domain failed: " << fi_strerror(-ret);
        goto err_fabric;
    }

    // active endpoint being used given this is a p2p architecture
    ret = fi_endpoint(domain, fi, &ep, nullptr);
    if (ret) {
        NIXL_ERROR << "fi_endpoint failed: " << fi_strerror(-ret);
        goto err_domain;
    }

    // bind the endpoint to a Completion Queue (CQ), the "data path"
    ret = fi_cq_open(domain, fi->cq_attr, &cq, nullptr);
    if (ret) {
        NIXL_ERROR << "fi_cq_open failed: " << fi_strerror(-ret);
        goto err_ep;
    }

    ret = fi_ep_bind(ep, &cq->fid, FI_SEND | FI_RECV);
    if (ret) {
        NIXL_ERROR << "fi_ep_bind to CQ failed: " << fi_strerror(-ret);
        goto err_cq;
    }

    // bind the endpoint to an Event Queue (EQ), the "control path"
    ret = fi_eq_open(fabric, fi->eq_attr, &eq, nullptr);
    if (ret) {
        NIXL_ERROR << "fi_eq_open failed: " << fi_strerror(-ret);
        goto err_cq;
    }

    ret = fi_ep_bind(ep, &eq->fid, FI_PE_BIND | FI_SOURCE | FI_RMA | FI_MSG);
    if (ret) {
        NIXL_ERROR << "fi_ep_bind to EQ failed: " << fi_strerror(-ret);
        goto err_eq;
    }

    ret = fi_listen(ep);
    if (ret) {
        NIXL_ERROR << "fi_listen failed: " << fi_strerror(-ret);
        goto err_ep;
    }

    ret = fi_enable(ep);
    if (ret) {
        NIXL_ERROR << "fi_enable failed: " << fi_strerror(-ret);
        goto err_ep;
    }

    size_t addrlen = 0;
    ret = fi_getname(&ep->fid, nullptr, &addrlen);
    if (ret != -FI_ETOOSMALL) {
        NIXL_ERROR << "fi_getname failed to get address length: " << fi_strerror(-ret);
        goto err_ep;
    }
    char *addr_buf = new char[addrlen];
    ret = fi_getname(&ep->fid, addr_buf, &addrlen);
    if (ret) {
        NIXL_ERROR << "fi_getname failed: " << fi_strerror(-ret);
        delete[] addr_buf;
        goto err_ep;
    }
    local_addr = std::string(addr_buf, addrlen);
    delete[] addr_buf;

    // cache provider use in connect()
    cached_provider_info = fi_dupinfo(fi);

    fi_freeinfo(hints);
    fi_freeinfo(info);

    eq_thread = std::thread(&nixlOFI_Engine::eq_event_loop, this);
    return;

err_eq:
    fi_close(&eq->fid);
err_cq:
    fi_close(&cq->fid);
err_ep:
    fi_close(&ep->fid);
err_domain:
    fi_close(&domain->fid);
err_fabric:
    fi_close(&fabric->fid);
err_getinfo:
    fi_freeinfo(hints);
    if (info) fi_freeinfo(info);
    setInitErr(NIXL_ERR_BACKEND);
}

nixlOFI_Engine::~nixlOFI_Engine() {
    eq_thread_stop = true;
    if (eq_thread.joinable()) {
        // wake up the EQ thread to ensure it exits
        fi_eq_read(eq, nullptr, 0, 0);
        eq_thread.join();
    }

    // close connected endpoints
    for (auto const& [key, val] : connected_eps) {
        fi_close(&val->fid);
    }

    if (ep) fi_close(&ep->fid);
    if (cq) fi_close(&cq->fid);
    if (eq) fi_close(&eq->fid);
    if (domain) fi_close(&domain->fid);
    if (fabric) fi_close(&fabric->fid);
    if (fi) fi_freeinfo(fi);
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
        mems.push_back(GAUDI_DEVICE_SEG);
    }
    if (hmem_ze_supported) {
        mems.push_back(INTEL_GPU_SEG);
    }
    return mems;
}

nixl_status_t nixlOFI_Engine::connect(const std::string &remote_agent) {
    std::lock_guard<std::mutex> lock(ep_lock);
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

    struct fi_info *hints = nullptr;
    struct fi_info *info = nullptr;
    int ret = 0;

    hints = fi_allocinfo();
    if (!hints) {
        NIXL_ERROR << "fi_allocinfo failed";
        return NIXL_ERR_BACKEND;
    }

    hints->ep_attr->type = FI_EP_RDM;
    hints->fabric_attr->prov_name = strdup(provider_name.c_str());

    ret = fi_getinfo(FI_VERSION(1, 0), remote_addr_str.c_str(), nullptr, 0, hints, &info);
    if (ret) {
        NIXL_ERROR << "fi_getinfo for remote agent failed: " << fi_strerror(-ret);
        fi_freeinfo(hints);
        return NIXL_ERR_BACKEND;
    }

    struct fi_info *remote_fi = info;
    if (!remote_fi) {
        NIXL_ERROR << "No provider info returned for remote agent";
        fi_freeinfo(hints);
        fi_freeinfo(info);
        return NIXL_ERR_BACKEND;
    }

    fid_ep *remote_ep = nullptr;
    ret = fi_endpoint(domain, remote_fi, &remote_ep, nullptr);
    if (ret) {
        NIXL_ERROR << "fi_endpoint for remote failed: " << fi_strerror(-ret);
        fi_freeinfo(hints);
        fi_freeinfo(info);
        return NIXL_ERR_BACKEND;
    }

    ret = fi_ep_bind(remote_ep, &cq->fid, FI_SEND | FI_RECV);
    if (ret) {
        NIXL_ERROR << "fi_ep_bind to CQ for remote failed: " << fi_strerror(-ret);
        fi_close(&remote_ep->fid);
        fi_freeinfo(hints);
        fi_freeinfo(info);
        return NIXL_ERR_BACKEND;
    }

    ret = fi_ep_bind(remote_ep, &eq->fid, FI_PE_BIND | FI_SOURCE | FI_RMA | FI_MSG);
    if (ret) {
        NIXL_ERROR << "fi_ep_bind to EQ for remote failed: " << fi_strerror(-ret);
        fi_close(&remote_ep->fid);
        fi_freeinfo(hints);
        fi_freeinfo(info);
        return NIXL_ERR_BACKEND;
    }

    ret = fi_enable(remote_ep);
    if (ret) {
        NIXL_ERROR << "fi_enable for remote failed: " << fi_strerror(-ret);
        fi_close(&remote_ep->fid);
        fi_freeinfo(hints);
        fi_freeinfo(info);
        return NIXL_ERR_BACKEND;
    }

    ret = fi_connect(remote_ep, remote_fi->dest_addr, local_agent_name.c_str(), local_agent_name.length() + 1);
    if (ret) {
        NIXL_ERROR << "fi_connect failed: " << fi_strerror(-ret);
        fi_close(&remote_ep->fid);
        free(hints->fabric_attr->prov_name);
        fi_freeinfo(hints);
        fi_freeinfo(info);
        return NIXL_ERR_BACKEND;
    }

    free(hints->fabric_attr->prov_name);
    fi_freeinfo(hints);
    fi_freeinfo(info);

    // wait for connection to complete via EQ
    struct fi_eq_cm_entry entry;
    uint32_t event;
    ssize_t n_events = fi_eq_read(eq, &event, &entry, 1, -1);
    if (n_events < 0) {
        NIXL_ERROR << "fi_eq_read failed during connect: " << fi_strerror(-n_events);
        fi_close(&remote_ep->fid);
        return NIXL_ERR_BACKEND;
    }
    if (event != FI_CONNECTED || entry.fid != &remote_ep->fid) {
        NIXL_ERROR << "Unexpected EQ event during connect: " << event;
        fi_close(&remote_ep->fid);
        return NIXL_ERR_BACKEND;
    }

    connected_eps[remote_agent] = remote_ep;

    NIXL_DEBUG << "OFI backend: Connected to " << remote_agent;
    return NIXL_SUCCESS;
}

nixl_status_t nixlOFI_Engine::disconnect(const std::string &remote_agent) {
    std::lock_guard<std::mutex> lock(ep_lock);
    auto it = connected_eps.find(remote_agent);
    if (it == connected_eps.end()) {
        NIXL_WARN << "OFI backend: No active connection to " << remote_agent;
        return NIXL_ERR_NOT_FOUND;
    }

    int ret = fi_close(&it->second->fid);
    if (ret) {
        NIXL_ERROR << "fi_close (remote_ep) failed: " << fi_strerror(-ret);
        return NIXL_ERR_BACKEND;
    }

    connected_eps.erase(it);
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
        ret = fi_mr_reg(domain, mem.addr, mem.len, FI_REMOTE_READ | FI_REMOTE_WRITE | FI_SEND | FI_RECV,
                            0, 0, 0, &ofi_meta->mr, nullptr);
    } else {
        struct fi_mr_attr mr_attr = {};
        struct iovec iov = {};

        iov.iov_base = mem.addr;
        iov.iov_len = mem.len;

        mr_attr.mr_iov = &iov;
        mr_attr.iov_count = 1;
        mr_attr.access = FI_REMOTE_READ | FI_REMOTE_WRITE | FI_SEND | FI_RECV;

        // Only host memory supported in core functionality
        NIXL_ERROR << "Non-host memory types not supported in core functionality";
        delete ofi_meta;
        return NIXL_ERR_NOT_SUPPORTED;

        ret = fi_mr_regattr(domain, &mr_attr, 0, &ofi_meta->mr);
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

    int ret = fi_close(&ofi_meta->mr->fid);
    if (ret) {
        NIXL_ERROR << "fi_close (mr) failed: " << fi_strerror(-ret);
        return NIXL_ERR_BACKEND;
    }

    delete ofi_meta;
    return NIXL_SUCCESS;
}

nixl_status_t nixlOFI_Engine::postXfer(const nixl_xfer_op_t &operation,
                                  const nixl_meta_dlist_t &local,
                                  const nixl_meta_dlist_t &remote,
                                  const std::string &remote_agent,
                                  nixlBackendReqH* &handle,
                                  const nixl_opt_b_args_t* opt_args) const {
    auto it = connected_eps.find(remote_agent);
    if (it == connected_eps.end()) {
        NIXL_ERROR << "OFI backend: Not connected to " << remote_agent;
        return NIXL_ERR_NOT_CONNECTED;
    }
    fid_ep *remote_ep = it->second;

    nixlOFI_Request *ofi_req = new nixlOFI_Request();
    if (!ofi_req) {
        return NIXL_ERR_BACKEND;
    }
    ofi_req->cq = cq;

    int ret = 0;
    for (size_t i = 0; i < local.descCount(); ++i) {
        const nixlMetaDesc &local_desc = local[i];
        const nixlMetaDesc &remote_desc = remote[i];

        nixlOFI_Metadata *local_meta = static_cast<nixlOFI_Metadata*>(local_desc.metadataP);
        nixlOFI_Metadata *remote_meta = static_cast<nixlOFI_Metadata*>(remote_desc.metadataP);

        struct iovec iov = {};
        iov.iov_base = local_desc.addr;
        iov.iov_len  = local_desc.len;

        struct fi_msg_rma msg = {};
        msg.msg_iov = &iov;
        msg.desc = &local_meta->desc;
        msg.iov_count = 1;
        msg.addr = 0; // For connected RDM, this is often FI_ADDR_UNSPEC (0)
        msg.context = &ofi_req->wr_id;

        struct fi_rma_iov rma_iov = {};
        if (operation == NIXL_READ || operation == NIXL_WRITE) {
            rma_iov.addr = (uint64_t)remote_desc.addr;
            rma_iov.len  = remote_desc.len;
            rma_iov.key  = fi_mr_key(remote_meta->mr);
            msg.rma_iov = &rma_iov;
            msg.rma_iov_count = 1;
        }

        switch (operation) {
            case NIXL_READ:
                ret = fi_readmsg(remote_ep, &msg, FI_COMPLETION);
                break;
            case NIXL_WRITE:
                ret = fi_writemsg(remote_ep, &msg, FI_COMPLETION);
                break;
            case NIXL_SEND:
                ret = fi_sendmsg(remote_ep, &msg, FI_COMPLETION);
                break;
            case NIXL_RECV:
                ret = fi_recvmsg(remote_ep, &msg, FI_COMPLETION);
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
                int accept_ret = fi_endpoint(domain, fi, &new_ep, nullptr);
                if (accept_ret) {
                    NIXL_ERROR << "fi_endpoint for accepted connection failed: " << fi_strerror(-accept_ret);
                    break;
                }
                accept_ret = fi_ep_bind(new_ep, &cq->fid, FI_SEND | FI_RECV);
                if (accept_ret) {
                    NIXL_ERROR << "fi_ep_bind to CQ for accepted connection failed: " << fi_strerror(-accept_ret);
                    fi_close(&new_ep->fid);
                    break;
                }
                accept_ret = fi_ep_bind(new_ep, &eq->fid, FI_PE_BIND | FI_SOURCE | FI_RMA | FI_MSG);
                if (accept_ret) {
                    NIXL_ERROR << "fi_ep_bind to EQ for accepted connection failed: " << fi_strerror(-accept_ret);
                    fi_close(&new_ep->fid);
                    break;
                }
                accept_ret = fi_accept(new_ep, entry.data);
                if (accept_ret) {
                    NIXL_ERROR << "fi_accept failed: " << fi_strerror(-accept_ret);
                    fi_close(&new_ep->fid);
                    break;
                }
                accept_ret = fi_enable(new_ep);
                if (accept_ret) {
                    NIXL_ERROR << "fi_enable for accepted connection failed: " << fi_strerror(-accept_ret);
                    fi_close(&new_ep->fid);
                    break;
                }

                std::string remote_agent_name;
                if (entry.data && entry.data_len > 0) {
                    remote_agent_name = std::string(static_cast<const char*>(entry.data), entry.data_len -1);
                } else {
                    NIXL_WARN << "No remote agent name received in connection request, connection may not be usable.";
                    // Create a placeholder name, as we cannot proceed without one.
                    // This connection will likely fail later if used.
                    remote_agent_name = "unknown_agent_" + std::to_string(reinterpret_cast<uintptr_t>(new_ep));
                }

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
            case FI_CMA_EVENT:
                NIXL_DEBUG << "FI_CMA_EVENT event received";
                // TODO: handle CMA events?
                break;
            case FI_EQ_ERR:
                NIXL_ERROR << "FI_EQ_ERR event received: " << fi_strerror(entry.err);
                break;
            default:
                NIXL_WARN << "Unhandled EQ event: " << event;
                break;
        }
    }
}
