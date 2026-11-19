/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// A notified WRITE whose posting fails partway, on whatever provider this host has (tcp, or verbs
// on an RDMA host), with faults injected under the plugin. The test defines fi_fabric(), which the
// plugin binds to instead of libfabric's, and from it hooks every domain and endpoint created:
// - fi_writemsg() can be failed (-FI_EIO) after a set number of writes have gone out, and holds a
//   write posted with FI_MORE until a post without it, as a provider that batches FI_MORE posts
//   (efa, cxi) may;
// - fi_senddata() records the control messages posted: transfer-error messages and
//   notifications, with their transfer IDs;
// - fi_cq_readfrom() withholds a transfer-error message's send completion for 100 ms, so the
//   initiator sees its transfer fail before the notification it owes may go out.
//
// The initiator runs without a progress thread, so a post fails inside postXfer(), which then
// returns NIXL_IN_PROG with the requests that went out still in flight; the target runs one, so it
// receives and answers without being polled (tcp progresses only when polled, on both sides). For
// each case it checks that the transfer ends with an error, and only once the notification it owes
// is posted: after its terminal result the initiator is not called again, as nixlAgent's cached
// getXferStatus() would not call it. It also checks that the handle counted exactly the
// writes that went out and saw all of them complete, that the transfer-error message was posted
// before the notification, and that the target received exactly one notification. Cases:
//   1. a failure after one successful post;
//   2. a striped descriptor partly submitted (skipped with a single rail);
//   3. a failure in one of the posting threads;
//   4. two failures in a row on one reused handle;
//   5. a failure with FI_MORE posts held until a flush: the transfer must still end.

#include <dlfcn.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_rma.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "libfabric_backend.h"
#include "test_utils.h"

using namespace std;
using namespace std::chrono;

/****************************************
 * Fault injection under the plugin
 *****************************************/

namespace {

// Copies a provider's ops table. The library at run time may be newer than the headers this test
// was built against, with a longer table: copy the provider's own size (every table starts with
// it), so the entries this test does not know about are kept.
template<typename T>
T *
copyOps(const T *real) {
    const size_t len = real->size > sizeof(T) ? real->size : sizeof(T);
    auto *ops = static_cast<T *>(calloc(1, len));
    memcpy(ops, real, real->size < len ? real->size : len);
    return ops;
}

struct EpOps {
    fi_ops_rma *real_rma = nullptr;
    fi_ops_msg *real_msg = nullptr;
    fi_ops_rma *hooked_rma = nullptr;
    fi_ops_msg *hooked_msg = nullptr;
};

struct HeldWrite {
    fid_ep *ep;
    fi_msg_rma msg;
    vector<iovec> iov;
    vector<void *> desc;
    vector<fi_rma_iov> rma_iov;
    uint64_t flags;
};

struct ControlPost {
    uint64_t type;
    uint16_t xfer_id;
};

mutex hook_mutex;
unordered_map<fid_fabric *, fi_ops_fabric *> real_fabric_ops;
unordered_map<fid_domain *, fi_ops_domain *> real_domain_ops;
unordered_map<fid_ep *, EpOps *> ep_ops;
vector<HeldWrite> held_writes;
// Every ops table this test installed. A provider may hand the same fabric (or domain, endpoint)
// back again, already hooked: wrapping it twice would record our own wrapper as the real ops.
unordered_set<const void *> installed_ops;
vector<ControlPost> control_posts;
atomic<int> endpoints_created{0};

// Writes allowed before every further write fails; -1 never fails.
atomic<long> writes_before_failure{-1};
atomic<long> writes_out{0}; // writes passed to the provider since the last reset
atomic<long> writes_held_total{0};
atomic<bool> hold_fi_more{false};

// A transfer-error message's send completion is withheld this long, so the initiator has reported
// the failure (or would have) before the notification may follow it.
constexpr milliseconds kErrorCompletionDelay(100);
unordered_set<void *> delayed_contexts; // send contexts of transfer-error messages

struct HeldCompletion {
    fid_cq *cq; // the CQ it came from, the only one it may be returned on
    fi_cq_data_entry entry;
    fi_addr_t src;
    steady_clock::time_point release;
};

vector<HeldCompletion> held_completions;
unordered_map<fid_cq *, fi_ops_cq *> real_cq_ops;

ssize_t
hooked_cq_readfrom(fid_cq *cq, void *buf, size_t count, fi_addr_t *src_addr) {
    fi_ops_cq *real;
    {
        const lock_guard<mutex> lock(hook_mutex);
        real = real_cq_ops.at(cq);
        // A withheld completion whose time has come is returned alone, by the CQ it came from.
        for (auto it = held_completions.begin(); it != held_completions.end(); ++it) {
            if (count > 0 && it->cq == cq && steady_clock::now() >= it->release) {
                static_cast<fi_cq_data_entry *>(buf)[0] = it->entry;
                if (src_addr != nullptr) {
                    src_addr[0] = it->src;
                }
                held_completions.erase(it);
                return 1;
            }
        }
    }
    ssize_t ret = real->readfrom(cq, buf, count, src_addr);
    if (ret <= 0) {
        return ret;
    }
    auto *entries = static_cast<fi_cq_data_entry *>(buf);
    const lock_guard<mutex> lock(hook_mutex);
    for (ssize_t i = 0; i < ret;) {
        if (delayed_contexts.erase(entries[i].op_context) == 0) {
            ++i;
            continue;
        }
        held_completions.push_back({cq,
                                    entries[i],
                                    src_addr != nullptr ? src_addr[i] : FI_ADDR_UNSPEC,
                                    steady_clock::now() + kErrorCompletionDelay});
        for (ssize_t j = i + 1; j < ret; ++j) {
            entries[j - 1] = entries[j];
            if (src_addr != nullptr) {
                src_addr[j - 1] = src_addr[j];
            }
        }
        --ret;
    }
    return ret > 0 ? ret : -FI_EAGAIN;
}

int
hooked_cq_open(fid_domain *domain, fi_cq_attr *attr, fid_cq **cq, void *context);

ssize_t
hooked_writemsg(fid_ep *ep, const fi_msg_rma *msg, uint64_t flags) {
    fi_ops_rma *real;
    {
        const lock_guard<mutex> lock(hook_mutex);
        real = ep_ops.at(ep)->real_rma;
    }
    // Claimed up front, so posting threads cannot both pass the limit; given back if not issued.
    const long allowed = writes_before_failure.load();
    if (writes_out.fetch_add(1) >= allowed && allowed >= 0) {
        --writes_out;
        return -FI_EIO;
    }
    if (hold_fi_more.load() && (flags & FI_MORE)) {
        // Accepted but not issued, until a post without FI_MORE on this endpoint flushes it.
        HeldWrite h{ep, *msg, {}, {}, {}, flags};
        h.iov.assign(msg->msg_iov, msg->msg_iov + msg->iov_count);
        if (msg->desc != nullptr) {
            h.desc.assign(msg->desc, msg->desc + msg->iov_count);
        }
        h.rma_iov.assign(msg->rma_iov, msg->rma_iov + msg->rma_iov_count);
        const lock_guard<mutex> lock(hook_mutex);
        held_writes.push_back(move(h));
        ++writes_held_total;
        return 0;
    }
    if (!(flags & FI_MORE)) {
        vector<HeldWrite> flush;
        {
            const lock_guard<mutex> lock(hook_mutex);
            for (auto it = held_writes.begin(); it != held_writes.end();) {
                if (it->ep == ep) {
                    flush.push_back(move(*it));
                    it = held_writes.erase(it);
                } else {
                    ++it;
                }
            }
        }
        for (HeldWrite &h : flush) {
            h.msg.msg_iov = h.iov.data();
            h.msg.desc = h.desc.empty() ? nullptr : h.desc.data();
            h.msg.rma_iov = h.rma_iov.data();
            real->writemsg(ep, &h.msg, h.flags & ~FI_MORE);
        }
    }
    const ssize_t ret = real->writemsg(ep, msg, flags);
    if (ret != 0) {
        --writes_out;
    }
    return ret;
}

ssize_t
hooked_senddata(fid_ep *ep,
                const void *buf,
                size_t len,
                void *desc,
                uint64_t data,
                fi_addr_t dest_addr,
                void *context) {
    fi_ops_msg *real;
    {
        const lock_guard<mutex> lock(hook_mutex);
        real = ep_ops.at(ep)->real_msg;
    }
    const ssize_t ret = real->senddata(ep, buf, len, desc, data, dest_addr, context);
    if (ret == 0) {
        const lock_guard<mutex> lock(hook_mutex);
        control_posts.push_back({NIXL_GET_MSG_TYPE_FROM_IMM(data),
                                 static_cast<uint16_t>(NIXL_GET_XFER_ID_FROM_IMM(data))});
        if (NIXL_GET_MSG_TYPE_FROM_IMM(data) == NIXL_LIBFABRIC_MSG_XFER_ERROR) {
            delayed_contexts.insert(context);
        }
    }
    return ret;
}

int
hooked_endpoint(fid_domain *domain, fi_info *info, fid_ep **ep, void *context) {
    fi_ops_domain *real;
    {
        const lock_guard<mutex> lock(hook_mutex);
        real = real_domain_ops.at(domain);
    }
    const int ret = real->endpoint(domain, info, ep, context);
    {
        // The plugin opens its CQ and endpoint before anything else on the domain, so the domain
        // hook is done: put the provider's own table back. A provider may recognize its domain
        // by that table (libfabric 2.0 tcp checks for its multiplexed domain at bind and enable).
        const lock_guard<mutex> lock(hook_mutex);
        domain->ops = real;
    }
    if (ret == 0) {
        const lock_guard<mutex> lock(hook_mutex);
        if (installed_ops.count((*ep)->rma) != 0) {
            return ret;
        }
        auto *ops = new EpOps();
        ops->real_rma = (*ep)->rma;
        ops->real_msg = (*ep)->msg;
        ops->hooked_rma = copyOps((*ep)->rma);
        ops->hooked_rma->writemsg = hooked_writemsg;
        ops->hooked_msg = copyOps((*ep)->msg);
        ops->hooked_msg->senddata = hooked_senddata;
        (*ep)->rma = ops->hooked_rma;
        (*ep)->msg = ops->hooked_msg;
        installed_ops.insert(ops->hooked_rma);
        installed_ops.insert(ops->hooked_msg);
        ep_ops[*ep] = ops;
        ++endpoints_created;
    }
    return ret;
}

int
hooked_domain(fid_fabric *fabric, fi_info *info, fid_domain **domain, void *context) {
    fi_ops_fabric *real;
    {
        const lock_guard<mutex> lock(hook_mutex);
        real = real_fabric_ops.at(fabric);
    }
    const int ret = real->domain(fabric, info, domain, context);
    if (ret == 0) {
        const lock_guard<mutex> lock(hook_mutex);
        if (installed_ops.count((*domain)->ops) == 0) {
            auto *ops = copyOps((*domain)->ops);
            real_domain_ops[*domain] = (*domain)->ops;
            ops->endpoint = hooked_endpoint;
            ops->cq_open = hooked_cq_open;
            (*domain)->ops = ops;
            installed_ops.insert(ops);
        }
    }
    return ret;
}

int
hooked_cq_open(fid_domain *domain, fi_cq_attr *attr, fid_cq **cq, void *context) {
    fi_ops_domain *real;
    {
        const lock_guard<mutex> lock(hook_mutex);
        real = real_domain_ops.at(domain);
    }
    const int ret = real->cq_open(domain, attr, cq, context);
    if (ret == 0 && attr->format == FI_CQ_FORMAT_DATA) {
        const lock_guard<mutex> lock(hook_mutex);
        if (installed_ops.count((*cq)->ops) == 0) {
            auto *ops = copyOps((*cq)->ops);
            real_cq_ops[*cq] = (*cq)->ops;
            ops->readfrom = hooked_cq_readfrom;
            (*cq)->ops = ops;
            installed_ops.insert(ops);
        }
    }
    return ret;
}

} // namespace

// The plugin's fi_fabric() binds here: the executable's definition comes first.
extern "C" int
fi_fabric(fi_fabric_attr *attr, fid_fabric **fabric, void *context) {
    using fabric_fn = int (*)(fi_fabric_attr *, fid_fabric **, void *);
    static const auto real_fabric =
        reinterpret_cast<fabric_fn>(dlvsym(RTLD_NEXT, "fi_fabric", "FABRIC_1.1"));
    if (real_fabric == nullptr) {
        cerr << "libfabric_post_failure_test: cannot find libfabric's fi_fabric" << endl;
        abort();
    }
    // A layered provider (ofi_rxm) opens its core fabric through fi_fabric() too, from inside the
    // plugin's call: hook only the outermost, the plugin's own fabric.
    static thread_local int depth = 0;
    ++depth;
    const int ret = real_fabric(attr, fabric, context);
    --depth;
    if (ret == 0 && depth == 0) {
        const lock_guard<mutex> lock(hook_mutex);
        if (installed_ops.count((*fabric)->ops) == 0) {
            auto *ops = copyOps((*fabric)->ops);
            real_fabric_ops[*fabric] = (*fabric)->ops;
            ops->domain = hooked_domain;
            (*fabric)->ops = ops;
            installed_ops.insert(ops);
        }
    }
    return ret;
}

/****************************************
 * Engines and buffers
 *****************************************/

namespace {

constexpr size_t kDescLen = 4096;
constexpr size_t kMaxDescs = 32;
constexpr size_t kStripedLen = 4 * NIXL_LIBFABRIC_DEFAULT_STRIPING_THRESHOLD;
constexpr size_t kBufLen = kMaxDescs * kDescLen + kStripedLen;
constexpr seconds kDeadline(15);

int failures = 0;

void
check(bool cond, const string &what) {
    cout << (cond ? "  PASS: " : "  FAIL: ") << what << endl;
    if (!cond) {
        ++failures;
    }
}

nixlLibfabricEngine *
createEngine(const string &name, nixl_b_params_t &params, bool progress_thread) {
    nixlBackendInitParams init;
    init.enableProgTh = progress_thread;
    init.pthrDelay = 100;
    init.localAgent = name;
    init.customParams = &params;
    init.type = "LIBFABRIC";
    auto *engine = new nixlLibfabricEngine(&init);
    nixl_exit_on_failure(!engine->getInitErr(), "Failed to initialize engine " + name);
    return engine;
}

struct Peers {
    nixlLibfabricEngine *a = nullptr; // initiator
    nixlLibfabricEngine *b = nullptr; // target
    vector<char> src, dst;
    nixlBackendMD *src_md = nullptr, *dst_md = nullptr, *remote_md = nullptr;
    int rails = 0;
};

nixlBackendMD *
registerBuf(nixlLibfabricEngine *engine, vector<char> &buf) {
    nixlBlobDesc desc;
    desc.addr = reinterpret_cast<uintptr_t>(buf.data());
    desc.len = buf.size();
    desc.devId = 0;
    nixlBackendMD *md = nullptr;
    nixl_exit_on_failure(engine->registerMem(desc, DRAM_SEG, md) == NIXL_SUCCESS,
                         "registerMem failed");
    return md;
}

void
setUp(Peers &p) {
    nixl_b_params_t params_a, params_b;
    // Posting threads for transfers of 8 descriptors or more (case 3); fewer post inline.
    params_a["num_threads"] = "2";
    params_a["split_batch_size"] = "8";
    const int before = endpoints_created.load();
    p.a = createEngine("FailA", params_a, /*progress_thread=*/false);
    p.rails = endpoints_created.load() - before;
    p.b = createEngine("FailB", params_b, /*progress_thread=*/true);

    string conn_a, conn_b;
    nixl_exit_on_failure(p.a->getConnInfo(conn_a) == NIXL_SUCCESS, "getConnInfo(A) failed");
    nixl_exit_on_failure(p.b->getConnInfo(conn_b) == NIXL_SUCCESS, "getConnInfo(B) failed");
    nixl_exit_on_failure(p.a->loadRemoteConnInfo("FailB", conn_b) == NIXL_SUCCESS,
                         "loadRemoteConnInfo(A, B) failed");
    nixl_exit_on_failure(p.b->loadRemoteConnInfo("FailA", conn_a) == NIXL_SUCCESS,
                         "loadRemoteConnInfo(B, A) failed");
    // Without a progress thread each side progresses only while connecting: connect both at once.
    nixl_status_t st_a = NIXL_ERR_BACKEND, st_b = NIXL_ERR_BACKEND;
    thread ta([&]() { st_a = p.a->connect("FailB"); });
    thread tb([&]() { st_b = p.b->connect("FailA"); });
    ta.join();
    tb.join();
    nixl_exit_on_failure(st_a == NIXL_SUCCESS && st_b == NIXL_SUCCESS, "connect failed");

    p.src.assign(kBufLen, 'a');
    p.dst.assign(kBufLen, 0);
    p.src_md = registerBuf(p.a, p.src);
    p.dst_md = registerBuf(p.b, p.dst);
    nixlBlobDesc remote;
    remote.addr = reinterpret_cast<uintptr_t>(p.dst.data());
    remote.len = p.dst.size();
    remote.devId = 0;
    nixl_exit_on_failure(p.b->getPublicData(p.dst_md, remote.metaInfo) == NIXL_SUCCESS,
                         "getPublicData failed");
    nixl_exit_on_failure(p.a->loadRemoteMD(remote, DRAM_SEG, "FailB", p.remote_md) == NIXL_SUCCESS,
                         "loadRemoteMD failed");
}

void
tearDown(Peers &p) {
    p.a->unloadMD(p.remote_md);
    p.a->deregisterMem(p.src_md);
    p.b->deregisterMem(p.dst_md);
    p.a->disconnect("FailB");
    p.b->disconnect("FailA");
    delete p.b;
    delete p.a;
}

// Descriptors [0, count) of kDescLen bytes, or one of kStripedLen bytes at the end of the buffers.
void
makeDescs(Peers &p, size_t count, bool striped, nixl_meta_dlist_t &l, nixl_meta_dlist_t &r) {
    for (size_t i = 0; i < count; ++i) {
        const size_t off = striped ? kMaxDescs * kDescLen : i * kDescLen;
        const size_t len = striped ? kStripedLen : kDescLen;
        nixlMetaDesc ld, rd;
        ld.addr = reinterpret_cast<uintptr_t>(p.src.data() + off);
        rd.addr = reinterpret_cast<uintptr_t>(p.dst.data() + off);
        ld.len = rd.len = len;
        ld.devId = rd.devId = 0;
        ld.metadataP = p.src_md;
        rd.metadataP = p.remote_md;
        l.addDesc(ld);
        r.addDesc(rd);
    }
}

// Collects the target's notifications carrying msg for `wait`. Only the target is polled: the
// initiator has reported its terminal result, after which an application (and nixlAgent's cached
// getXferStatus()) need not call it again, so what it owes the target must already be on its way.
size_t
countNotifs(Peers &p, const string &msg, milliseconds wait) {
    size_t n = 0;
    const auto until = steady_clock::now() + wait;
    while (steady_clock::now() < until) {
        notif_list_t notifs;
        p.b->getNotifs(notifs);
        for (const auto &[agent, text] : notifs) {
            n += (agent == "FailA" && text == msg) ? 1 : 0;
        }
        this_thread::sleep_for(milliseconds(2));
    }
    return n;
}

/**
 * Posts one notified WRITE on handle (prepared on first use) with every write after
 * writes_allowed failing, and checks how it ends.
 */
void
failedWrite(Peers &p,
            const string &label,
            size_t count,
            bool striped,
            long writes_allowed,
            nixlBackendReqH *&handle,
            const string &msg) {
    cout << label << endl;
    nixl_meta_dlist_t l(DRAM_SEG), r(DRAM_SEG);
    makeDescs(p, count, striped, l, r);
    nixl_opt_b_args_t opt;
    opt.notifMsg = msg;
    opt.hasNotif = true;
    if (handle == nullptr) {
        nixl_exit_on_failure(p.a->prepXfer(NIXL_WRITE, l, r, "FailB", handle, &opt) == NIXL_SUCCESS,
                             "prepXfer failed");
    }
    {
        const lock_guard<mutex> lock(hook_mutex);
        control_posts.clear();
    }
    writes_out = 0;
    writes_before_failure = writes_allowed;

    nixl_status_t st = p.a->postXfer(NIXL_WRITE, l, r, "FailB", handle, &opt);
    const long went_out = writes_out.load();
    writes_before_failure = -1;
    check(st == NIXL_IN_PROG,
          "postXfer() returns NIXL_IN_PROG with " + to_string(went_out) + " write(s) in flight");
    check(went_out == writes_allowed, "exactly the allowed writes went out");

    // The target progresses too, so it receives the writes, the failure notice and the
    // notification while the initiator polls.
    size_t notifs = 0;
    const auto deadline = steady_clock::now() + kDeadline;
    while (st == NIXL_IN_PROG && steady_clock::now() < deadline) {
        st = p.a->checkXfer(handle);
        notif_list_t got;
        p.b->getNotifs(got);
        for (const auto &[agent, text] : got) {
            notifs += (agent == "FailA" && text == msg) ? 1 : 0;
        }
    }
    check(st != NIXL_IN_PROG && st != NIXL_SUCCESS,
          "checkXfer() reports the failure (status " + to_string(st) + ") within " +
              to_string(kDeadline.count()) + " s");

    auto *h = static_cast<nixlLibfabricBackendH *>(handle);
    check(h->get_submitted_requests_count() == static_cast<size_t>(went_out),
          "the handle counts exactly the " + to_string(went_out) + " request(s) that went out");
    check(h->get_completed_requests_count() == h->get_submitted_requests_count(),
          "every counted request completed before the failure was reported");

    // The notification follows the failure notice's completion; a duplicate would follow shortly
    // after the original.
    notifs += countNotifs(p, msg, milliseconds(1500));
    check(notifs == 1,
          "the target received exactly one notification (got " + to_string(notifs) + ")");

    size_t errors = 0, notif_posts = 0;
    bool error_first = false;
    {
        const lock_guard<mutex> lock(hook_mutex);
        for (const ControlPost &c : control_posts) {
            if (c.xfer_id != h->post_xfer_id) {
                continue;
            }
            if (c.type == NIXL_LIBFABRIC_MSG_XFER_ERROR) {
                ++errors;
            } else if (c.type == NIXL_LIBFABRIC_MSG_NOTIFICTION) {
                error_first = error_first || (notif_posts == 0 && errors > 0);
                ++notif_posts;
            }
        }
    }
    check(errors == 1, "one transfer-error message was posted (got " + to_string(errors) + ")");
    check(notif_posts == 1,
          "one notification was posted (got " + to_string(notif_posts) +
              "), after it: " + (error_first ? "yes" : "no"));
    check(error_first, "the transfer-error message was posted before the notification");
}

} // namespace

int
main() {
    Peers p;
    setUp(p);
    cout << "libfabric_post_failure_test: " << p.rails << " rail(s)" << endl;

    nixlBackendReqH *h1 = nullptr;
    failedWrite(p, "1. Failure after one successful post", 4, false, 1, h1, "fail-1");
    p.a->releaseReqH(h1);

    if (p.rails > 1) {
        nixlBackendReqH *h2 = nullptr;
        failedWrite(p, "2. Striped descriptor partly submitted", 1, true, 1, h2, "fail-2");
        p.a->releaseReqH(h2);
    } else {
        cout << "2. SKIP: a single rail does not stripe" << endl;
    }

    nixlBackendReqH *h3 = nullptr;
    failedWrite(p, "3. Failure in a posting thread", 16, false, 5, h3, "fail-3");
    p.a->releaseReqH(h3);

    nixlBackendReqH *h4 = nullptr;
    failedWrite(p, "4a. Reused handle, first failure", 4, false, 2, h4, "fail-4");
    failedWrite(p, "4b. Reused handle, second failure", 4, false, 1, h4, "fail-4");
    p.a->releaseReqH(h4);

    hold_fi_more = true;
    nixlBackendReqH *h5 = nullptr;
    failedWrite(p, "5. Failure with FI_MORE posts held until a flush", 6, false, 3, h5, "fail-5");
    p.a->releaseReqH(h5);
    hold_fi_more = false;
    {
        const lock_guard<mutex> lock(hook_mutex);
        check(held_writes.empty(), "no write is left held for a flush");
    }
    check(writes_held_total.load() == 0, "no write was posted with FI_MORE");

    // Afterwards a transfer on the same peers succeeds.
    cout << "6. A transfer after the failures" << endl;
    {
        nixl_meta_dlist_t l(DRAM_SEG), r(DRAM_SEG);
        makeDescs(p, 4, false, l, r);
        nixl_opt_b_args_t opt;
        opt.notifMsg = "ok-6";
        opt.hasNotif = true;
        nixlBackendReqH *h = nullptr;
        nixl_exit_on_failure(p.a->prepXfer(NIXL_WRITE, l, r, "FailB", h, &opt) == NIXL_SUCCESS,
                             "prepXfer failed");
        nixl_status_t st = p.a->postXfer(NIXL_WRITE, l, r, "FailB", h, &opt);
        const auto deadline = steady_clock::now() + kDeadline;
        size_t notifs = 0;
        while (st == NIXL_IN_PROG && steady_clock::now() < deadline) {
            st = p.a->checkXfer(h);
            notif_list_t got;
            p.b->getNotifs(got);
            notifs += got.size();
        }
        notifs += countNotifs(p, "ok-6", milliseconds(500));
        check(st == NIXL_SUCCESS, "it completes");
        check(notifs == 1, "and delivers one notification");
        p.a->releaseReqH(h);
    }

    tearDown(p);
    if (failures != 0) {
        cout << "libfabric_post_failure_test: " << failures << " check(s) FAILED" << endl;
        return 1;
    }
    cout << "libfabric_post_failure_test OK" << endl;
    return 0;
}
