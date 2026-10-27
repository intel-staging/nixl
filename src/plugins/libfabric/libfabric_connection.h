/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 Amazon.com, Inc. and affiliates.
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
#ifndef NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_CONNECTION_H
#define NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_CONNECTION_H

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "libfabric/libfabric_rail.h"

/** Multi-rail connection metadata for remote agents */
struct nixlLibfabricConnection : public nixlBackendConnMD {
    nixlLibfabricConnection(const std::string &remote_agent, size_t agent_index);

    /** Transition to CONNECTED state. */
    nixl_status_t
    establish();

    /** Transition to DISCONNECTED state. Returns success if already disconnected. */
    nixl_status_t
    disconnect();

    size_t agent_index_; // Unique agent identifier in agent_names vector
    std::string remoteAgent_; // Remote agent name
    std::unordered_map<size_t, std::vector<fi_addr_t>>
        rail_remote_addr_list_; // Rail libfabric addresses. key=rail id.
    std::vector<char *> src_ep_names_; // Rail endpoint names
    std::atomic<ConnectionState> overall_state_; // Current connection state

    // Set once the peer holds our endpoints in its AV, so it can attribute our traffic.
    // establishConnection() blocks on handshake_cv_ until this or handshake_rejected_ is set.
    std::atomic<bool> handshake_received_{false};
    // Set when the peer's protocol version is incompatible, so establishConnection() fails with
    // NIXL_ERR_MISMATCH. rejected_peer_proto_ver_ is written under handshake_mutex_ first.
    std::atomic<bool> handshake_rejected_{false};
    uint16_t rejected_peer_proto_ver_ = 0;
    std::mutex handshake_mutex_;
    std::condition_variable handshake_cv_;

    /**
     * @brief Write tags, for providers that report no source address for remote writes
     * (see providerNeedsWriteTag()).
     *
     * local_write_tag_ is the tag we handed this peer in our handshake, mapped on our rails to its
     * agent index. peer_write_tag_ is the tag it handed us, stamped in the imm_data of our writes
     * to it. 0 means no tag. Each tag has a generation, bumped on reuse and stamped in the SEQ_ID
     * bits, so writes from the tag's previous holder are dropped; peer_write_gen_ is 0 when the
     * peer sent none, and then no generation is stamped.
     */
    uint8_t local_write_tag_ = 0;
    uint8_t local_write_gen_ = 0;
    std::atomic<uint8_t> peer_write_tag_{0};
    std::atomic<uint8_t> peer_write_gen_{0};
};

#endif // NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_CONNECTION_H
