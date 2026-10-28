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

#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#include "libfabric_backend.h"
#include "serdes/serdes.h"
#include "test_utils.h"

using namespace std;
using namespace std::chrono;

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

    static bool
    handshakeBuffered(nixlLibfabricEngine &engine, const string &remote_agent) {
        const lock_guard<mutex> lock(engine.pending_handshake_mutex_);
        return engine.pending_inbound_handshakes_.count(remote_agent) != 0;
    }
};

using TestAccess = nixlLibfabricEngineTestAccess;

static nixlLibfabricEngine *
createEngine(const string &name) {
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

int
main() {
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
