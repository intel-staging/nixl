/*
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
 * Rail-level handling of refused posts and CQ errors, with libfabric mocked out. By default
 * (post_eagain_timeout 0) a refused post is retried until the destination takes it, however long
 * that is. With a timeout set, a destination that refuses posts with -FI_EAGAIN fails them all once
 * it has refused for that long, not each after a timeout of its own; an accepted post clears that.
 * A CQ error reaches its own request instead of failing whoever progressed the CQ.
 */

#include "libfabric/libfabric_rail_manager.h"
#include "libfabric/libfabric_common.h"
#include "common/nixl_log.h"
#include "libfabric_mock_stubs.h"

#include <chrono>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

static const size_t NUM_FAKE_RAILS = 1;

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

// --- Fake CQ: empty, or one error entry ---

static bool cq_error_pending = false;
static struct fi_cq_err_entry cq_error = {};

static ssize_t
fi_cq_readfrom_stub(struct fid_cq *, void *, size_t, fi_addr_t *) {
    return cq_error_pending ? -FI_EAVAIL : -FI_EAGAIN;
}

static ssize_t
fi_cq_readerr_stub(struct fid_cq *, struct fi_cq_err_entry *buf, uint64_t) {
    if (!cq_error_pending) {
        return -FI_EAGAIN;
    }
    cq_error_pending = false;
    *buf = cq_error;
    return 1;
}

// --- fi_writemsg that refuses one destination ---

static const fi_addr_t kRefusing = 7;
static int refuse_remaining = 0; // -1: refuse forever
static std::chrono::steady_clock::time_point refuse_until{}; // also refused until then

static bool
refusesNow(fi_addr_t dest_addr) {
    if (dest_addr != kRefusing) {
        return false;
    }
    if (std::chrono::steady_clock::now() < refuse_until) {
        return true;
    }
    if (refuse_remaining != 0) {
        if (refuse_remaining > 0) {
            --refuse_remaining;
        }
        return true;
    }
    return false;
}

static ssize_t after_refusals = 0; // what the post returns once the refusals are used up

static ssize_t
refusing_writemsg(struct fid_ep *, const struct fi_msg_rma *msg, uint64_t) {
    if (refusesNow(msg->addr)) {
        return -FI_EAGAIN;
    }
    return msg->addr == kRefusing ? after_refusals : 0;
}

// fi_senddata that refuses kRefusing the same way
static ssize_t
refusing_senddata(struct fid_ep *,
                  const void *,
                  size_t,
                  void *,
                  uint64_t,
                  fi_addr_t dest_addr,
                  void *) {
    return refusesNow(dest_addr) ? -FI_EAGAIN : 0;
}

static size_t recvs_posted = 0;

static ssize_t
counting_recvmsg(struct fid_ep *, const struct fi_msg *, uint64_t) {
    ++recvs_posted;
    return 0;
}

static const std::chrono::seconds kTimeout(1);

#define TEST_ASSERT(cond, msg)                                                           \
    do {                                                                                 \
        if (!(cond)) {                                                                   \
            std::cerr << "FAIL: " << (msg) << " [" << __FILE__ << ":" << __LINE__ << "]" \
                      << std::endl;                                                      \
            return 1;                                                                    \
        }                                                                                \
    } while (0)

static nixl_status_t
postTo(nixlLibfabricRail &rail, nixlLibfabricReq *req, fi_addr_t dest) {
    static char buf[64];
    return rail.postWrite(buf, sizeof(buf), nullptr, 0, dest, 0x1000, 0x42, req, 0, false);
}

// --- Tests ---

// With the default post_eagain_timeout (0) a refused post never fails for being refused: a write
// refused for longer than the timeout the other tests set is retried until taken, and a send under
// PostOnceScope keeps returning NIXL_IN_PROG and goes out once the destination takes it.
static int
testDefaultRetriesThroughRefusal(nixlLibfabricRail &rail) {
    NIXL_INFO << "  testDefaultRetriesThroughRefusal";
    using namespace std::chrono;
    const auto refusal = kTimeout + milliseconds(500);
    nixlLibfabricReq *req = rail.allocateDataRequest(nixlLibfabricReq::WRITE, 1);
    TEST_ASSERT(req != nullptr, "allocated a data request");
    fi_ep_rma_ops_stub.writemsg = refusing_writemsg;
    auto t0 = steady_clock::now();
    refuse_until = t0 + refusal;
    TEST_ASSERT(postTo(rail, req, kRefusing) == NIXL_SUCCESS,
                "by default a refused write is retried until the destination takes it");
    TEST_ASSERT(steady_clock::now() - t0 >= refusal, "the write waited out the refusal");
    fi_ep_rma_ops_stub.writemsg = fi_ep_writemsg_stub;
    rail.releaseRequest(req);

    nixlLibfabricReq *send = rail.allocateControlRequest(64, 1);
    TEST_ASSERT(send != nullptr, "allocated a control request");
    send->buffer_size = 8;
    fi_ep_msg_ops_stub.senddata = refusing_senddata;
    {
        const nixlLibfabricRail::PostOnceScope post_once;
        t0 = steady_clock::now();
        refuse_until = t0 + refusal;
        nixl_status_t status = NIXL_IN_PROG;
        int refused = 0;
        while (status == NIXL_IN_PROG && steady_clock::now() - t0 < 3 * refusal) {
            status = rail.postSend(0, kRefusing, send);
            refused += status == NIXL_IN_PROG;
            if (status == NIXL_IN_PROG) {
                std::this_thread::sleep_for(milliseconds(50));
            }
        }
        TEST_ASSERT(status == NIXL_SUCCESS && refused > 0,
                    "by default a send refused under the scope stays pending, never failed, and "
                    "goes out once the destination takes it");
        TEST_ASSERT(steady_clock::now() - t0 >= refusal, "the send waited out the refusal");
    }
    fi_ep_msg_ops_stub.senddata = fi_ep_senddata_stub;
    rail.releaseRequest(send);
    refuse_until = {};
    return 0;
}

// A destination that keeps refusing fails its first post after the timeout and every later one at
// once; other destinations are unaffected, and an accepted post clears the record.
static int
testRefusedDestinationFailsTogether(nixlLibfabricRail &rail) {
    NIXL_INFO << "  testRefusedDestinationFailsTogether";
    using namespace std::chrono;
    nixlLibfabricReq *req = rail.allocateDataRequest(nixlLibfabricReq::WRITE, 1);
    TEST_ASSERT(req != nullptr, "allocated a data request");
    fi_ep_rma_ops_stub.writemsg = refusing_writemsg;

    refuse_remaining = -1;
    auto t0 = steady_clock::now();
    TEST_ASSERT(postTo(rail, req, kRefusing) == NIXL_ERR_BACKEND,
                "a post the destination keeps refusing fails");
    TEST_ASSERT(steady_clock::now() - t0 >= kTimeout,
                "the first refused post waits for the timeout");
    t0 = steady_clock::now();
    TEST_ASSERT(postTo(rail, req, kRefusing) == NIXL_ERR_BACKEND,
                "a later post to the refusing destination fails");
    TEST_ASSERT(steady_clock::now() - t0 < milliseconds(500),
                "a later post to the refusing destination fails at once");
    TEST_ASSERT(postTo(rail, req, kRefusing + 1) == NIXL_SUCCESS,
                "another destination is unaffected");

    // Accepted again, then refused briefly: the record was cleared, so it retries and succeeds
    refuse_remaining = 0;
    TEST_ASSERT(postTo(rail, req, kRefusing) == NIXL_SUCCESS, "the destination accepts again");
    refuse_remaining = 3;
    TEST_ASSERT(postTo(rail, req, kRefusing) == NIXL_SUCCESS,
                "after an accepted post, a briefly refused one is retried, not failed");

    // A refused post that then fails with another error leaves no record behind, so a brief
    // refusal long after is retried, not failed at once
    refuse_remaining = 2;
    after_refusals = -FI_EIO;
    TEST_ASSERT(postTo(rail, req, kRefusing) == NIXL_ERR_BACKEND, "the post fails with -FI_EIO");
    after_refusals = 0;
    std::this_thread::sleep_for(kTimeout + milliseconds(200));
    refuse_remaining = 1;
    TEST_ASSERT(postTo(rail, req, kRefusing) == NIXL_SUCCESS,
                "a refused post that failed otherwise left a stale record");

    fi_ep_rma_ops_stub.writemsg = fi_ep_writemsg_stub;
    rail.releaseRequest(req);
    return 0;
}

// Under PostOnceScope (the progress thread's sends) a refused send is tried once and returns
// NIXL_IN_PROG, but the destination's refusal timer still runs: once it has refused for the
// timeout, the send fails, and every later one at once, so a queued notification to a peer that
// keeps refusing cannot keep its transfer in progress forever. An accepted send clears it.
static int
testPostOnceRefusalTimesOut(nixlLibfabricRail &rail) {
    NIXL_INFO << "  testPostOnceRefusalTimesOut";
    using namespace std::chrono;
    nixlLibfabricReq *req = rail.allocateControlRequest(64, 1);
    TEST_ASSERT(req != nullptr, "allocated a control request");
    req->buffer_size = 8;
    fi_ep_msg_ops_stub.senddata = refusing_senddata;
    {
        const nixlLibfabricRail::PostOnceScope post_once;
        refuse_remaining = -1;
        const auto t0 = steady_clock::now();
        TEST_ASSERT(rail.postSend(0, kRefusing, req) == NIXL_IN_PROG,
                    "a refused send under the scope is not retried");
        nixl_status_t status = NIXL_IN_PROG;
        while (status == NIXL_IN_PROG && steady_clock::now() - t0 < 3 * kTimeout) {
            std::this_thread::sleep_for(milliseconds(50));
            status = rail.postSend(0, kRefusing, req);
        }
        const auto waited = steady_clock::now() - t0;
        TEST_ASSERT(status == NIXL_ERR_BACKEND,
                    "a send the destination keeps refusing fails under the scope too");
        TEST_ASSERT(waited >= kTimeout && waited < kTimeout + milliseconds(500),
                    "it fails once the destination has refused for the timeout");
        const auto t1 = steady_clock::now();
        TEST_ASSERT(rail.postSend(0, kRefusing, req) == NIXL_ERR_BACKEND &&
                        steady_clock::now() - t1 < milliseconds(100),
                    "a later send to the refusing destination fails at once");

        refuse_remaining = 0;
        TEST_ASSERT(rail.postSend(0, kRefusing, req) == NIXL_SUCCESS,
                    "the destination accepts again");
        refuse_remaining = 1;
        TEST_ASSERT(rail.postSend(0, kRefusing, req) == NIXL_IN_PROG,
                    "after an accepted send, a refused one is not failed at once");
        TEST_ASSERT(rail.postSend(0, kRefusing, req) == NIXL_SUCCESS,
                    "and is accepted when the destination takes it");
    }
    fi_ep_msg_ops_stub.senddata = fi_ep_senddata_stub;
    rail.releaseRequest(req);
    return 0;
}

// A CQ error goes to its own request's callback and counts as handled; one that belongs to no
// request is returned.
static int
testCqErrorGoesToItsRequest(nixlLibfabricRail &rail) {
    NIXL_INFO << "  testCqErrorGoesToItsRequest";
    nixlLibfabricReq *req = rail.allocateDataRequest(nixlLibfabricReq::WRITE, 1);
    TEST_ASSERT(req != nullptr, "allocated a data request");
    nixl_status_t seen = NIXL_SUCCESS;
    req->completion_callback = [&seen](nixl_status_t status) { seen = status; };

    cq_error = {};
    cq_error.op_context = &req->ctx;
    cq_error.err = FI_ECANCELED;
    cq_error_pending = true;
    TEST_ASSERT(rail.progressCompletionQueue() == NIXL_SUCCESS,
                "a CQ error delivered to its request is not a progress error");
    TEST_ASSERT(seen == NIXL_ERR_BACKEND, "the request's callback got the error");

    cq_error = {};
    cq_error.op_context = nullptr;
    cq_error.err = FI_ECANCELED;
    cq_error_pending = true;
    TEST_ASSERT(rail.progressCompletionQueue() == NIXL_ERR_BACKEND,
                "a CQ error that belongs to no request is returned");

    // A control send has no callback to tell: its error is returned
    nixlLibfabricReq *send = rail.allocateControlRequest(64, 1);
    TEST_ASSERT(send != nullptr, "allocated a control request");
    cq_error = {};
    cq_error.op_context = &send->ctx;
    cq_error.err = FI_EIO;
    cq_error_pending = true;
    TEST_ASSERT(rail.progressCompletionQueue() == NIXL_ERR_BACKEND,
                "a control send's CQ error is returned");

    // A receive posted through postRecv() that completes with an error is replaced, unless it was
    // cancelled
    fi_ep_msg_ops_stub.recvmsg = counting_recvmsg;
    for (const int err : {FI_EIO, FI_ECANCELED}) {
        nixlLibfabricReq *recv = rail.allocateControlRequest(64, 2);
        TEST_ASSERT(recv != nullptr, "allocated a receive request");
        TEST_ASSERT(rail.postRecv(recv) == NIXL_SUCCESS, "posted a receive");
        recvs_posted = 0;
        cq_error = {};
        cq_error.op_context = &recv->ctx;
        cq_error.err = err;
        cq_error_pending = true;
        TEST_ASSERT(rail.progressCompletionQueue() == NIXL_SUCCESS,
                    "a receive's CQ error is handled");
        TEST_ASSERT(recvs_posted == (err == FI_ECANCELED ? 0u : 1u),
                    err == FI_ECANCELED ? "a cancelled receive is not replaced" :
                                          "a failed receive is replaced");
    }

    // A control request that was a receive is a send again once reallocated, so its error is
    // returned; a CQ error flagged FI_RECV is a receive's whatever the request's type
    nixlLibfabricReq *reused = rail.allocateControlRequest(64, 3);
    TEST_ASSERT(reused != nullptr, "allocated a control request");
    TEST_ASSERT(reused->operation_type == nixlLibfabricReq::SEND,
                "a reallocated control request is a send");
    cq_error = {};
    cq_error.op_context = &reused->ctx;
    cq_error.err = FI_EIO;
    cq_error.flags = FI_RECV | FI_MSG;
    cq_error_pending = true;
    recvs_posted = 0;
    TEST_ASSERT(rail.progressCompletionQueue() == NIXL_SUCCESS && recvs_posted == 1,
                "a CQ error flagged FI_RECV is handled as a receive's");
    fi_ep_msg_ops_stub.recvmsg = fi_ep_recvmsg_stub;
    return 0;
}

int
main() {
    NIXL_INFO << "=== Refused post and CQ error test ===";
    cq_ops_stub.size = sizeof(fi_ops_cq);
    cq_ops_stub.readfrom = fi_cq_readfrom_stub;
    cq_ops_stub.readerr = fi_cq_readerr_stub;

    nixlLibfabricRail rail("efa_0", "efa", 93, FI_HMEM_SYSTEM, true);
    int res;
    if ((res = testDefaultRetriesThroughRefusal(rail)) != 0) {
        return res;
    }
    // The rest run with an explicit timeout
    rail.setPostEagainTimeout(kTimeout);
    if ((res = testCqErrorGoesToItsRequest(rail)) != 0 ||
        (res = testRefusedDestinationFailsTogether(rail)) != 0 ||
        (res = testPostOnceRefusalTimesOut(rail)) != 0) {
        return res;
    }
    NIXL_INFO << "=== All refused post and CQ error tests PASSED ===";
    return 0;
}
