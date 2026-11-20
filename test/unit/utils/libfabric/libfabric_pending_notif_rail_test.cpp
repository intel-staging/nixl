/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 Amazon.com, Inc. and affiliates.
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
 * Rail-level pieces of pending-notification tracking, with libfabric mocked out (the mocks are
 * those of src_addr_resolution_test.cpp): XFER_IDs, writes without remote CQ data, and a
 * reconnect taking over its old index's source addresses.
 */

#include "libfabric/libfabric_rail_manager.h"
#include "libfabric/libfabric_common.h"
#include "common/nixl_log.h"
#include "libfabric_mock_stubs.h"

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

// Number of fake EFA devices to create
static const size_t NUM_FAKE_RAILS = 4;

// --- Unconditional __wrap_* functions ---

extern "C" int
__wrap_numa_max_node() {
    return 1;
}

extern "C" int
__wrap_numa_num_configured_nodes() {
    return 2;
}

extern "C" int
__wrap_fi_getinfo(uint32_t /*version*/,
                  const char * /*node*/,
                  const char * /*service*/,
                  uint64_t /*flags*/,
                  const struct fi_info * /*hints*/,
                  struct fi_info **info) {
    *info = mock_fi_info_chain(NUM_FAKE_RAILS, 100ull * NIXL_LIBFABRIC_GIGA);
    return 0;
}

extern "C" int
__wrap_fi_fabric(struct fi_fabric_attr * /*attr*/, struct fid_fabric **fabric, void * /*context*/) {
    *fabric = mock_fabric_create();
    return 0;
}

// --- Fake completion queue ---
//
// Tests push completions plus their source addresses, then call
// progressCompletionQueue() and inspect what the rail's callbacks saw.

struct FakeCompletion {
    struct fi_cq_data_entry entry;
    fi_addr_t src_addr;
};

static std::vector<FakeCompletion> pending_completions;
static size_t completions_consumed = 0;

static ssize_t
fi_cq_readfrom_stub(struct fid_cq * /*cq*/, void *buf, size_t count, fi_addr_t *src_addr) {
    if (completions_consumed >= pending_completions.size()) {
        return -FI_EAGAIN;
    }
    auto *entries = static_cast<struct fi_cq_data_entry *>(buf);
    size_t n = 0;
    while (n < count && completions_consumed < pending_completions.size()) {
        const FakeCompletion &fc = pending_completions[completions_consumed++];
        entries[n] = fc.entry;
        // A caller that forgets to size the src_addr array to `count`, or that mixes up the
        // two arrays, shows up here.
        if (src_addr != nullptr) {
            src_addr[n] = fc.src_addr;
        }
        ++n;
    }
    return static_cast<ssize_t>(n);
}

static ssize_t
fi_cq_readerr_stub(struct fid_cq * /*cq*/, struct fi_cq_err_entry * /*buf*/, uint64_t /*flags*/) {
    return -FI_EAGAIN;
}

static void
queue_reset() {
    pending_completions.clear();
    completions_consumed = 0;
}

// --- Test helpers ---

#define TEST_ASSERT(cond, msg)                                                           \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::cerr << "FAIL: " << (msg) << " [" << __FILE__ << ":" << __LINE__ << "]" \
                      << std::endl;                                                      \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

// Distinct fake endpoint-name blobs, one per (agent, remote rail) pair.
static std::vector<std::array<char, LF_EP_NAME_MAX_LEN>>
makeEndpoints(const std::string &agent, size_t num_rails) {
    std::vector<std::array<char, LF_EP_NAME_MAX_LEN>> eps(num_rails);
    for (size_t r = 0; r < num_rails; ++r) {
        eps[r].fill(0);
        const std::string name = agent + ":rail" + std::to_string(r);
        std::memcpy(eps[r].data(), name.c_str(), std::min(name.size(), eps[r].size() - 1));
    }
    return eps;
}

// --- Tests ---

// getNextXferId() wraps the 16-bit counter without ever returning NIXL_UNTRACKED_XFER_ID.
static int
testXferIdSkipsUntracked() {
    NIXL_INFO << "  testXferIdSkipsUntracked";
    for (uint32_t i = 0; i < 2 * (NIXL_XFER_ID_MASK + 1u); ++i) {
        TEST_ASSERT(LibfabricUtils::getNextXferId() != NIXL_UNTRACKED_XFER_ID,
                    "getNextXferId() never returns the untracked XFER_ID");
    }
    return 0;
}

// fi_av_insert() that fails on its fail_av_insert_at-th call (counting from 1), else as the stub
static size_t av_insert_calls = 0;
static size_t fail_av_insert_at = 0;

static int
failing_av_insert(struct fid_av *av,
                  const void *addr,
                  size_t count,
                  fi_addr_t *fi_addr,
                  uint64_t flags,
                  void *context) {
    if (++av_insert_calls == fail_av_insert_at) {
        return -FI_EINVAL;
    }
    return fi_av_insert_stub(av, addr, count, fi_addr, flags, context);
}

// A reconnect gets the same fi_addrs back from the address vector and its new index takes over
// each of them on every rail. One that fails partway hands each address back to its previous
// index: the old index, or none for a new one.
static int
testReconnectReplacesIndex(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testReconnectReplacesIndex";
    mock_av_reset();

    const size_t num_rails = mgr.getNumRails();
    const auto eps = makeEndpoints("peerY", num_rails);
    std::unordered_map<size_t, std::vector<fi_addr_t>> fi_addrs, fi_addrs2;
    std::vector<char *> ep_names, ep_names2;
    TEST_ASSERT(mgr.insertAllAddresses(eps, 77, fi_addrs, ep_names) == NIXL_SUCCESS,
                "insertAllAddresses succeeded");

    // Fails on the first endpoint of the last rail, after the other rails were remapped
    av_ops_stub.insert = failing_av_insert;
    av_insert_calls = 0;
    fail_av_insert_at = (num_rails - 1) * num_rails + 1;
    const auto eps_fresh = makeEndpoints("peerZ", num_rails);
    TEST_ASSERT(mgr.insertAllAddresses(eps, 78, fi_addrs2, ep_names2, 77) != NIXL_SUCCESS,
                "the failing reconnect reports the failure");
    for (size_t rail_id = 0; rail_id < num_rails; ++rail_id) {
        for (fi_addr_t a : fi_addrs.at(rail_id)) {
            TEST_ASSERT(mgr.getRail(rail_id).resolveSourceAddress(a) == 77u,
                        "a failed reconnect hands its addresses back to the old index");
        }
    }
    av_insert_calls = 0;
    TEST_ASSERT(mgr.insertAllAddresses(eps_fresh, 90, fi_addrs2, ep_names2) != NIXL_SUCCESS,
                "the failing fresh connect reports the failure");
    TEST_ASSERT(mgr.getRail(0).resolveSourceAddress(fi_addrs2.at(0).at(0)) ==
                    nixlLibfabricRail::kUnknownAgentIdx,
                "a failed fresh connect leaves its addresses unmapped");
    av_ops_stub.insert = fi_av_insert_stub;

    TEST_ASSERT(mgr.insertAllAddresses(eps, 79, fi_addrs2, ep_names2, 77) == NIXL_SUCCESS,
                "reconnect insertAllAddresses succeeded");
    for (size_t rail_id = 0; rail_id < num_rails; ++rail_id) {
        TEST_ASSERT(fi_addrs2.at(rail_id) == fi_addrs.at(rail_id),
                    "the reconnect gets the same fi_addrs back");
        for (fi_addr_t a : fi_addrs2.at(rail_id)) {
            TEST_ASSERT(mgr.getRail(rail_id).resolveSourceAddress(a) == 79u,
                        "the new index takes over every address on every rail");
        }
    }
    return 0;
}

// A write of a transfer without a notification is posted without remote CQ data; the others with.
static int
testUntrackedWriteCarriesNoCqData() {
    NIXL_INFO << "  testUntrackedWriteCarriesNoCqData";
    nixlLibfabricRail rail("efa_0", "efa", 91, FI_HMEM_SYSTEM, true);
    nixlLibfabricReq *req = rail.allocateDataRequest(nixlLibfabricReq::WRITE, 1);
    TEST_ASSERT(req != nullptr, "allocated a data request");
    char buf[64] = {};
    for (const bool tracked : {true, false}) {
        mock_writemsg_last_flags = 0;
        TEST_ASSERT(
            rail.postWrite(buf,
                           sizeof(buf),
                           nullptr,
                           tracked ? NIXL_MAKE_IMM_DATA(NIXL_LIBFABRIC_MSG_TRANSFER, 7, 1) : 0,
                           0,
                           0x1000,
                           0x42,
                           req,
                           0,
                           tracked) == NIXL_SUCCESS,
            "postWrite succeeded");
        TEST_ASSERT(((mock_writemsg_last_flags & FI_REMOTE_CQ_DATA) != 0) == tracked,
                    tracked ? "a tracked write carries remote CQ data" :
                              "a write of an untracked transfer carries no remote CQ data");
    }
    rail.releaseRequest(req);
    return 0;
}

// A provider with FI_RMA_EVENT (cxi) raises a target event even for a write without remote CQ
// data, whose data field is then undefined (here a stale TRANSFER value): it is not counted.
static int
testRemoteWriteWithoutCqDataIgnored(nixlLibfabricRailManager &mgr) {
    NIXL_INFO << "  testRemoteWriteWithoutCqDataIgnored";
    mock_av_reset();
    queue_reset();
    nixlLibfabricRail &rail = mgr.getRail(0);
    const auto eps = makeEndpoints("peerC", 1);
    fi_addr_t addr = FI_ADDR_UNSPEC;
    TEST_ASSERT(rail.insertAddress(eps[0].data(), 5, &addr) == NIXL_SUCCESS, "insert peer");

    std::vector<uint16_t> seen;
    rail.setXferIdCallback([&seen](uint64_t imm_data, uint32_t) {
        seen.push_back(static_cast<uint16_t>(NIXL_GET_XFER_ID_FROM_IMM(imm_data)));
    });
    for (const uint64_t flags : {uint64_t(FI_REMOTE_WRITE | FI_RMA),
                                 uint64_t(FI_REMOTE_WRITE | FI_REMOTE_CQ_DATA | FI_RMA)}) {
        FakeCompletion fc{};
        fc.entry.flags = flags;
        fc.entry.len = 4096;
        fc.entry.data = NIXL_MAKE_IMM_DATA(NIXL_LIBFABRIC_MSG_TRANSFER, 55, 0);
        fc.src_addr = addr;
        pending_completions.push_back(fc);
    }
    TEST_ASSERT(rail.progressCompletionQueue() == NIXL_SUCCESS, "both completions processed");
    TEST_ASSERT(seen.size() == 1 && seen[0] == 55, "only the write with remote CQ data is counted");
    rail.setXferIdCallback(nullptr);
    return 0;
}

int
main() {
    NIXL_INFO << "=== Pending-notification rail test ===";
    cq_ops_stub.size = sizeof(fi_ops_cq);
    cq_ops_stub.readfrom = fi_cq_readfrom_stub;
    cq_ops_stub.readerr = fi_cq_readerr_stub;

    nixlLibfabricRailManager mgr(0);
    TEST_ASSERT(mgr.getNumRails() == NUM_FAKE_RAILS,
                "expected " + std::to_string(NUM_FAKE_RAILS) + " rails");

    int res;
    if ((res = testXferIdSkipsUntracked()) != 0 || (res = testReconnectReplacesIndex(mgr)) != 0 ||
        (res = testUntrackedWriteCarriesNoCqData()) != 0 ||
        (res = testRemoteWriteWithoutCqDataIgnored(mgr)) != 0) {
        return res;
    }
    NIXL_INFO << "=== All pending-notification rail tests PASSED ===";
    return 0;
}
