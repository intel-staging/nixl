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
#include "libfabric/libfabric_rail_manager.h"

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

    /**
     * @brief Which of this peer's rails each local rail should post to. Indexed by local rail id.
     *
     * Rails used to be paired by position, which silently assumes both agents enumerated their rails
     * onto the same fabrics in the same order. Nothing guarantees that -- domain names are host-local
     * and the cabling need not follow them -- and when it is wrong a rail posts to a peer NIC on
     * another segment, where the address is unreachable rather than invalid, so the write retries
     * -FI_EAGAIN forever. This maps each local rail to a peer rail that is actually on its segment.
     *
     * Entries hold NIXL_LF_NO_PEER_RAIL where no match was found, which puts that rail back on index
     * order. Empty when the provider's address format cannot be read at all.
     */
    std::vector<size_t> peer_rail_for_local_rail_;

    /**
     * @brief Which of this peer's rails each local rail is known not to reach, from the same
     *        computeRailPeering() call. Lets a transfer leave out a local rail that reaches none of
     *        the peer rails a buffer is registered on; see nixlLibfabricRailUnreachable.
     */
    nixlLibfabricRailUnreachable rail_unreachable_;

    /**
     * @brief Local rail to send notifications and handshakes from. Defaults to rail 0.
     *
     * Control messages must *arrive* on the peer's rail 0, because that is the only rail with the
     * notification and handshake receive callbacks installed. Rail 0 locally is not necessarily able
     * to reach the peer's rail 0 -- on a host pair whose rails are cabled across, it reaches a
     * different segment and the send retries -FI_EAGAIN forever. So this is the local rail that
     * @ref peer_rail_for_local_rail_ pairs with peer rail 0, which keeps the arrival side fixed at 0
     * while making the departure side reachable.
     */
    size_t control_rail_ = 0;

    /**
     * @brief Write tags, for providers whose remote-write completions carry no usable source address.
     *
     * A target must know which peer each of its incoming writes came from, to credit it to that
     * peer's pending notification. It normally learns that from the completion's source address,
     * but ofi_rxm reports none for a write (see providerNeedsWriteTag()). On such a provider each
     * agent hands every peer a tag in its handshake, and the peer stamps it on the writes it sends
     * back. 0 means no tag: the target resolves the source address as usual.
     *
     * local_write_tag_ is the tag we handed this peer, mapped on our rails to its agent index and
     * released on disconnect. peer_write_tag_ is the one it handed us, stamped on our writes to it;
     * set from its handshake (or to local_write_tag_ for the self-connection), before any write.
     *
     * A tag is reused once its connection is gone, but its old holder is not told and may keep
     * writing. So each tag also has a generation, bumped on every reuse and stamped in the SEQ_ID
     * bits; the target drops a write whose generation is not the current one. local_write_gen_ and
     * peer_write_gen_ pair with the two tags above; peer_write_gen_ stays 0 for a peer that
     * predates generations, and then nothing is stamped or checked.
     */
    uint8_t local_write_tag_ = 0;
    uint8_t local_write_gen_ = 0;
    std::atomic<uint8_t> peer_write_tag_{0};
    std::atomic<uint8_t> peer_write_gen_{0};
    std::atomic<ConnectionState> overall_state_; // Current connection state

    // Set once the peer holds our endpoints in its AV, so it can attribute our traffic.
    // establishConnection() blocks on handshake_cv_ until this or handshake_rejected_ is set.
    std::atomic<bool> handshake_received_{false};
    // Set when the peer's protocol version is incompatible, so establishConnection() fails with
    // NIXL_ERR_MISMATCH. rejected_peer_proto_ver_ is written under handshake_mutex_ first.
    std::atomic<bool> handshake_rejected_{false};
    uint16_t rejected_peer_proto_ver_ = 0;

    // Whether our own handshake to this peer has been posted, or is queued to be (deferred, or
    // waiting out -FI_EAGAIN). Cleared when that fails for good, so an inbound handshake can tell
    // that the peer never heard from us on this connection and answer it. Mutable because the
    // send path holds the connection const.
    mutable std::atomic<bool> handshake_sent_{false};
    std::mutex handshake_mutex_;
    std::condition_variable handshake_cv_;
};

#endif // NIXL_SRC_PLUGINS_LIBFABRIC_LIBFABRIC_CONNECTION_H
