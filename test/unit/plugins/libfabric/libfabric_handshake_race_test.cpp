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

// A first-contact handshake whose connection a user thread registers between handleHandshake()'s
// lookup and its own create, on whatever provider this host has. The handshake must be applied to
// that connection, its reply sent, and the handshake it buffered cleared, rather than lost.
//
// Neither engine sends a handshake of its own: both register the other with
// createConnectionFromInfo(), which sends nothing, so the only handshake A can receive from B is
// the reply under test.
//
// Also: one peer whose handshake replies the provider keeps refusing (-FI_EAGAIN, with the default
// post_eagain_timeout of 0, so they are never failed) does not hold up replies to other peers.
// The target's endpoints are hooked through an fi_fabric() of this test's own, which refuses its
// handshake sends to that one peer.

#include <dlfcn.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>

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
#include "serdes/serdes.h"
#include "test_utils.h"

using namespace std;
using namespace std::chrono;

/****************************************
 * Refusing one destination's handshake sends
 *****************************************/

namespace {

// Copies a provider's ops table at the provider's own size (the run-time library may be newer
// than the headers this test was built against, with a longer table)
template<typename T>
T *
copyOps(const T *real) {
    const size_t len = real->size > sizeof(T) ? real->size : sizeof(T);
    auto *ops = static_cast<T *>(calloc(1, len));
    memcpy(ops, real, real->size < len ? real->size : len);
    return ops;
}

mutex hook_mutex;
unordered_map<fid_fabric *, fi_ops_fabric *> real_fabric_ops;
unordered_map<fid_domain *, fi_ops_domain *> real_domain_ops;
unordered_map<fid_ep *, fi_ops_msg *> real_msg_ops;
unordered_map<fid_ep *, string> ep_owner; // engine whose construction opened the endpoint
unordered_set<const void *> installed_ops; // tables this test installed, never wrapped again
string creating; // the engine being constructed

// Handshake sends from refuse_owner's endpoints to refuse_addr are refused; attempts, and sends the
// provider accepted, are counted per destination of refuse_owner's endpoints. A provider that
// connects on demand may itself refuse the first sends to a peer.
string refuse_owner;
atomic<fi_addr_t> refuse_addr{FI_ADDR_UNSPEC};
unordered_map<fi_addr_t, int> handshake_attempts;
unordered_map<fi_addr_t, int> handshake_accepted;

ssize_t
hooked_senddata(fid_ep *ep,
                const void *buf,
                size_t len,
                void *desc,
                uint64_t data,
                fi_addr_t dest_addr,
                void *context) {
    fi_ops_msg *real;
    bool refuse = false;
    bool counted = false;
    {
        const lock_guard<mutex> lock(hook_mutex);
        real = real_msg_ops.at(ep);
        if (NIXL_GET_MSG_TYPE_FROM_IMM(data) == NIXL_LIBFABRIC_MSG_HANDSHAKE &&
            ep_owner[ep] == refuse_owner && !refuse_owner.empty()) {
            ++handshake_attempts[dest_addr];
            refuse = dest_addr == refuse_addr.load();
            counted = true;
        }
    }
    if (refuse) {
        return -FI_EAGAIN;
    }
    const ssize_t ret = real->senddata(ep, buf, len, desc, data, dest_addr, context);
    if (counted && ret == 0) {
        const lock_guard<mutex> lock(hook_mutex);
        ++handshake_accepted[dest_addr];
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
    const lock_guard<mutex> lock(hook_mutex);
    // A provider may recognize its domain by its own table (libfabric 2.0 tcp): restore it
    domain->ops = real;
    if (ret == 0 && installed_ops.count((*ep)->msg) == 0) {
        real_msg_ops[*ep] = (*ep)->msg;
        auto *ops = copyOps((*ep)->msg);
        ops->senddata = hooked_senddata;
        (*ep)->msg = ops;
        installed_ops.insert(ops);
        ep_owner[*ep] = creating;
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
            real_domain_ops[*domain] = (*domain)->ops;
            auto *ops = copyOps((*domain)->ops);
            ops->endpoint = hooked_endpoint;
            (*domain)->ops = ops;
            installed_ops.insert(ops);
        }
    }
    return ret;
}

} // namespace

// The plugin's fi_fabric() binds here: the executable's definition comes first
extern "C" int
fi_fabric(fi_fabric_attr *attr, fid_fabric **fabric, void *context) {
    using fabric_fn = int (*)(fi_fabric_attr *, fid_fabric **, void *);
    static const auto real_fabric =
        reinterpret_cast<fabric_fn>(dlvsym(RTLD_NEXT, "fi_fabric", "FABRIC_1.1"));
    if (real_fabric == nullptr) {
        cerr << "libfabric_handshake_race_test: cannot find libfabric's fi_fabric" << endl;
        abort();
    }
    // A layered provider (ofi_rxm) opens its core fabric through fi_fabric() too: hook only the
    // outermost, the plugin's own
    static thread_local int depth = 0;
    ++depth;
    const int ret = real_fabric(attr, fabric, context);
    --depth;
    if (ret == 0 && depth == 0) {
        const lock_guard<mutex> lock(hook_mutex);
        if (installed_ops.count((*fabric)->ops) == 0) {
            real_fabric_ops[*fabric] = (*fabric)->ops;
            auto *ops = copyOps((*fabric)->ops);
            ops->domain = hooked_domain;
            (*fabric)->ops = ops;
            installed_ops.insert(ops);
        }
    }
    return ret;
}

// Reaches the engine's private handshake handling
class nixlLibfabricEngineTestAccess {
public:
    static void
    setLookupMissHook(nixlLibfabricEngine &engine, function<void(const string &)> hook) {
        engine.handshake_lookup_miss_hook_ = std::move(hook);
    }

    static nixl_status_t
    registerWithoutHandshake(nixlLibfabricEngine &engine,
                             const string &remote_agent,
                             const string &conn_info) {
        shared_ptr<nixlLibfabricConnection> conn;
        return engine.createConnectionFromInfo(remote_agent, conn_info, conn);
    }

    static void
    handleHandshake(nixlLibfabricEngine &engine, const string &payload) {
        engine.handleHandshake(payload);
    }

    static bool
    handshakeReceived(nixlLibfabricEngine &engine, const string &remote_agent) {
        const lock_guard<mutex> lock(engine.connection_state_mutex_);
        const auto it = engine.connections_.find(remote_agent);
        return it != engine.connections_.end() &&
            it->second->handshake_received_.load(memory_order_acquire);
    }

    // Queue a handshake reply to @p remote_agent, as handleHandshake() does with a progress thread
    static void
    queueReply(nixlLibfabricEngine &engine, const string &remote_agent) {
        shared_ptr<nixlLibfabricConnection> conn;
        {
            const lock_guard<mutex> lock(engine.connection_state_mutex_);
            conn = engine.connections_.at(remote_agent);
        }
        engine.replyHandshake(conn, "test reply");
    }

    // The fi_addr of @p remote_agent's first endpoint on rail 0
    static fi_addr_t
    firstRemoteAddr(nixlLibfabricEngine &engine, const string &remote_agent) {
        const lock_guard<mutex> lock(engine.connection_state_mutex_);
        return engine.connections_.at(remote_agent)->rail_remote_addr_list_.at(0).at(0);
    }

    static bool
    handshakeBuffered(nixlLibfabricEngine &engine, const string &remote_agent) {
        const lock_guard<mutex> lock(engine.pending_handshake_mutex_);
        return engine.pending_inbound_handshakes_.count(remote_agent) != 0;
    }
};

using TestAccess = nixlLibfabricEngineTestAccess;

static nixlLibfabricEngine *
createEngine(const string &name) {
    {
        const lock_guard<mutex> lock(hook_mutex);
        creating = name;
    }
    nixlBackendInitParams init;
    nixl_b_params_t custom_params;
    init.enableProgTh = true;
    init.pthrDelay = 100;
    init.localAgent = name;
    init.customParams = &custom_params;
    init.type = "LIBFABRIC";
    auto *engine = new nixlLibfabricEngine(&init);
    nixl_exit_on_failure(!engine->getInitErr(), "Failed to initialize engine " + name);
    return engine;
}

// A first-contact handshake from @p sender, as sendHandshakeTo() builds it
static string
firstContactHandshake(const string &sender, const string &sender_conn_info) {
    const uint16_t proto_ver = static_cast<uint16_t>(NIXL_LIBFABRIC_PROTO_VERSION);
    const uint8_t has_conn = 1;
    nixlSerDes sd;
    sd.addBuf(NIXL_HANDSHAKE_TAG_VER, &proto_ver, sizeof(proto_ver));
    sd.addStr(NIXL_HANDSHAKE_TAG_NAME, sender);
    sd.addBuf(NIXL_HANDSHAKE_TAG_HAS_CONN, &has_conn, sizeof(has_conn));
    sd.addStr(NIXL_HANDSHAKE_TAG_CONN, sender_conn_info);
    return sd.exportStr();
}

// A target T replies to three peers queued in the order R, H1, H2. The provider keeps refusing
// T's handshake sends to R, and post_eagain_timeout is the default 0, so R's reply stays queued.
// H1 and H2 must still get theirs, each exactly once, while R's keeps being tried.
static void
runRefusingPeerScenario() {
    const string t_name = "AgentT";
    const vector<string> peers = {"AgentR", "AgentH1", "AgentH2"};
    auto *t = createEngine(t_name);
    vector<nixlLibfabricEngine *> p;
    string conn_t;
    nixl_exit_on_failure(t->getConnInfo(conn_t) == NIXL_SUCCESS, "getConnInfo(T) failed");
    for (const string &name : peers) {
        p.push_back(createEngine(name));
        string conn_p;
        nixl_exit_on_failure(p.back()->getConnInfo(conn_p) == NIXL_SUCCESS, "getConnInfo failed");
        nixl_exit_on_failure(
            TestAccess::registerWithoutHandshake(*t, name, conn_p) == NIXL_SUCCESS &&
                TestAccess::registerWithoutHandshake(*p.back(), t_name, conn_t) == NIXL_SUCCESS,
            "registering " + name + " failed");
    }
    const fi_addr_t r_addr = TestAccess::firstRemoteAddr(*t, peers[0]);
    {
        const lock_guard<mutex> lock(hook_mutex);
        refuse_owner = t_name;
        refuse_addr = r_addr;
        handshake_attempts.clear();
        handshake_accepted.clear();
    }
    for (const string &name : peers) {
        TestAccess::queueReply(*t, name);
    }

    const auto deadline = steady_clock::now() + seconds(10);
    auto healthy_done = [&] {
        return TestAccess::handshakeReceived(*p[1], t_name) &&
            TestAccess::handshakeReceived(*p[2], t_name);
    };
    while (!healthy_done() && steady_clock::now() < deadline) {
        this_thread::sleep_for(milliseconds(1));
    }
    this_thread::sleep_for(milliseconds(200)); // a few more passes for R
    int r_attempts, h1_sent, h2_sent;
    const bool r_replied_while_refused = TestAccess::handshakeReceived(*p[0], t_name);
    {
        const lock_guard<mutex> lock(hook_mutex);
        r_attempts = handshake_attempts[r_addr];
        h1_sent = handshake_accepted[TestAccess::firstRemoteAddr(*t, peers[1])];
        h2_sent = handshake_accepted[TestAccess::firstRemoteAddr(*t, peers[2])];
        refuse_owner.clear(); // R's reply may go out now
    }
    nixl_exit_on_failure(healthy_done(),
                         "a peer whose replies are refused held up the other peers' replies (" +
                             to_string(r_attempts) + " attempts to it, " + to_string(h1_sent) +
                             " and " + to_string(h2_sent) + " sent to the others)");
    nixl_exit_on_failure(h1_sent == 1 && h2_sent == 1,
                         "a healthy peer's reply was not sent exactly once (" + to_string(h1_sent) +
                             ", " + to_string(h2_sent) + ")");
    nixl_exit_on_failure(r_attempts > 1, "the refused reply was not tried again");
    nixl_exit_on_failure(!r_replied_while_refused, "the refused peer got a reply while refused");
    // Once accepted, R's reply goes out too
    const auto r_deadline = steady_clock::now() + seconds(10);
    while (!TestAccess::handshakeReceived(*p[0], t_name) && steady_clock::now() < r_deadline) {
        this_thread::sleep_for(milliseconds(1));
    }
    nixl_exit_on_failure(TestAccess::handshakeReceived(*p[0], t_name),
                         "the refused reply did not go out once the provider took it");
    cout << "libfabric_handshake_race_test OK: replies to two peers went out while a third's was "
            "refused "
         << r_attempts << " times, then that one went out too" << endl;

    for (size_t i = 0; i < peers.size(); ++i) {
        t->disconnect(peers[i]);
        p[i]->disconnect(t_name);
    }
    delete t; // the replying side's progress thread goes first
    for (auto *e : p) {
        delete e;
    }
}

int
main() {
    runRefusingPeerScenario();
    const string a_name = "AgentA", b_name = "AgentB";
    auto *a = createEngine(a_name);
    auto *b = createEngine(b_name);

    string conn_a, conn_b;
    nixl_exit_on_failure(a->getConnInfo(conn_a) == NIXL_SUCCESS, "getConnInfo(A) failed");
    nixl_exit_on_failure(b->getConnInfo(conn_b) == NIXL_SUCCESS, "getConnInfo(B) failed");
    nixl_exit_on_failure(TestAccess::registerWithoutHandshake(*a, b_name, conn_b) == NIXL_SUCCESS,
                         "A could not register B");

    // B's lookup for A misses; a user thread then registers A before B buffers the handshake
    bool hook_ran = false;
    TestAccess::setLookupMissHook(*b, [&](const string &peer) {
        if (peer == a_name && !hook_ran) {
            hook_ran = true;
            nixl_exit_on_failure(TestAccess::registerWithoutHandshake(*b, a_name, conn_a) ==
                                     NIXL_SUCCESS,
                                 "the racing registration of A failed");
        }
    });
    TestAccess::handleHandshake(*b, firstContactHandshake(a_name, conn_a));
    TestAccess::setLookupMissHook(*b, nullptr);

    nixl_exit_on_failure(hook_ran, "the lookup did not miss, so the race was not exercised");
    nixl_exit_on_failure(TestAccess::handshakeReceived(*b, a_name),
                         "the handshake was not applied to the connection registered meanwhile");
    nixl_exit_on_failure(!TestAccess::handshakeBuffered(*b, a_name),
                         "the handshake was left buffered");

    // B's reply is the only handshake A can get from B
    const auto deadline = steady_clock::now() + seconds(10);
    while (!TestAccess::handshakeReceived(*a, b_name) && steady_clock::now() < deadline) {
        this_thread::sleep_for(milliseconds(1));
    }
    nixl_exit_on_failure(TestAccess::handshakeReceived(*a, b_name),
                         "B did not reply to the handshake");

    cout << "libfabric_handshake_race_test OK: a handshake whose connection was registered "
            "between lookup and buffering was applied, replied to, and not left buffered"
         << endl;

    b->disconnect(a_name);
    a->disconnect(b_name);
    // B is the one replying: its progress thread must be gone before A frees its buffers
    delete b;
    delete a;
    return 0;
}
