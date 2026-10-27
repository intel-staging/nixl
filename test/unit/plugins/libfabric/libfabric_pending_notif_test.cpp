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

// A target's pending notifications, on whatever provider this host has (meson also runs it with
// FI_PROVIDER=tcp, whose target counts writes): WRITEs without a notification leave nothing to
// track; which stale entries are dropped; entries and source mappings surviving a disconnect; a
// reconnect carrying entries over; and the stale-age setting.

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "libfabric_backend.h"
#include "test_utils.h"

using namespace std;
using namespace std::chrono;

// Reaches the engine's private test helpers
class nixlLibfabricEngineTestAccess {
public:
    static size_t
    reapStale(nixlLibfabricEngine &engine, steady_clock::time_point now) {
        return engine.reapStalePendingNotifications(now);
    }

    static size_t
    pendingCount(nixlLibfabricEngine &engine, uint32_t sender_peer_idx) {
        const lock_guard<mutex> lock(engine.receiver_tracking_mutex_);
        size_t n = 0;
        for (const auto &entry : engine.pending_notifications_) {
            n += nixlLibfabricEngine::pendingKeySender(entry.first) == sender_peer_idx;
        }
        return n;
    }

    static optional<uint32_t>
    agentIndex(const nixlLibfabricEngine &engine, const string &remote_agent) {
        const lock_guard<mutex> lock(engine.connection_state_mutex_);
        const auto it = engine.connections_.find(remote_agent);
        if (it == engine.connections_.end()) {
            return nullopt;
        }
        return static_cast<uint32_t>(it->second->agent_index_);
    }

    static void
    maybeReap(nixlLibfabricEngine &engine, bool drained) {
        engine.maybeReapStalePendingNotifications(drained);
    }

    static seconds
    writesAge(const nixlLibfabricEngine &engine) {
        return engine.stale_writes_age_;
    }

    static bool
    notifiesAfterCompletion(const nixlLibfabricEngine &engine) {
        return engine.notify_after_completion_;
    }

    static seconds
    staleAge(const nixlLibfabricEngine &engine) {
        return engine.stale_pending_age_;
    }

    static bool
    isLive(nixlLibfabricEngine &engine, uint32_t sender_peer_idx) {
        const lock_guard<mutex> lock(engine.receiver_tracking_mutex_);
        return engine.live_senders_.count(sender_peer_idx) != 0;
    }

    // Writes counted for (sender, xfer_id), or -1 if there is no entry
    static long
    receivedWrites(nixlLibfabricEngine &engine, uint32_t sender_peer_idx, uint16_t xfer_id) {
        const lock_guard<mutex> lock(engine.receiver_tracking_mutex_);
        const auto it = engine.pending_notifications_.find(
            nixlLibfabricEngine::makePendingKey(sender_peer_idx, xfer_id));
        return it == engine.pending_notifications_.end() ?
            -1L :
            static_cast<long>(it->second.received_completions);
    }

    // Fragment 0 of an @p of_fragments notification from @p agent expecting @p expected writes,
    // as processNotification() would store it, without checking whether it is complete
    static void
    addNotification(nixlLibfabricEngine &engine,
                    uint32_t sender_peer_idx,
                    uint16_t xfer_id,
                    uint32_t expected,
                    const string &agent,
                    const string &msg,
                    uint16_t of_fragments = 1) {
        const lock_guard<mutex> lock(engine.receiver_tracking_mutex_);
        auto [it, inserted] = engine.pending_notifications_.try_emplace(
            nixlLibfabricEngine::makePendingKey(sender_peer_idx, xfer_id), xfer_id);
        it->second.message_fragments.assign(of_fragments, "");
        it->second.message_fragments[0] = agent + msg;
        it->second.expected_msg_fragments = of_fragments;
        it->second.received_msg_fragments = 1;
        it->second.expected_completions = expected;
        it->second.agent_name_length = static_cast<uint16_t>(agent.size());
        it->second.total_message_length = static_cast<uint32_t>(agent.size() + msg.size());
        it->second.last_activity = steady_clock::now();
    }

    // The fi_addr of @p remote_agent's first endpoint on rail 0
    static fi_addr_t
    firstRemoteAddr(const nixlLibfabricEngine &engine, const string &remote_agent) {
        const lock_guard<mutex> lock(engine.connection_state_mutex_);
        return engine.connections_.at(remote_agent)->rail_remote_addr_list_.at(0).at(0);
    }

    static uint32_t
    resolveOnRail0(nixlLibfabricEngine &engine, fi_addr_t fi_addr) {
        return engine.rail_manager_.getRail(0).resolveSourceAddress(fi_addr);
    }
};

using TestAccess = nixlLibfabricEngineTestAccess;

static const size_t kLen = 1 << 20;
static const int kIters = 50;
static const seconds kNotifWait(10);
static const int kProgressThreadDelayUs = 100;

static nixlLibfabricEngine *
createEngine(const string &name, bool progress_thread = true) {
    nixlBackendInitParams init;
    nixl_b_params_t custom_params;

    init.enableProgTh = progress_thread;
    init.pthrDelay = kProgressThreadDelayUs;
    init.localAgent = name;
    init.customParams = &custom_params;
    init.type = "LIBFABRIC";

    auto *engine = new nixlLibfabricEngine(&init);
    nixl_exit_on_failure(!engine->getInitErr(), "Failed to initialize engine " + name);
    return engine;
}

static nixlBackendMD *
registerBuf(nixlLibfabricEngine *engine, void *addr) {
    nixlBlobDesc desc;
    desc.addr = reinterpret_cast<uintptr_t>(addr);
    desc.len = kLen;
    desc.devId = 0;
    nixlBackendMD *md = nullptr;
    nixl_exit_on_failure(engine->registerMem(desc, DRAM_SEG, md) == NIXL_SUCCESS,
                         "registerMem failed");
    return md;
}

// Polls the target's notifications until one arrives or kNotifWait passes. getNotifs() returns
// NIXL_IN_PROG when there are none.
static notif_list_t
waitNotifs(nixlLibfabricEngine *target) {
    notif_list_t notifs;
    const auto deadline = steady_clock::now() + kNotifWait;
    while (notifs.empty() && steady_clock::now() < deadline) {
        nixl_exit_on_failure(target->getNotifs(notifs) >= NIXL_SUCCESS, "getNotifs failed");
        if (notifs.empty()) {
            this_thread::sleep_for(microseconds(50));
        }
    }
    return notifs;
}

// Connects a and b both ways; returns a's connection info
static string
connectPair(nixlLibfabricEngine *a,
            const string &a_name,
            nixlLibfabricEngine *b,
            const string &b_name) {
    string conn_a, conn_b;
    nixl_exit_on_failure(a->getConnInfo(conn_a) == NIXL_SUCCESS, "getConnInfo(A) failed");
    nixl_exit_on_failure(b->getConnInfo(conn_b) == NIXL_SUCCESS, "getConnInfo(B) failed");
    nixl_exit_on_failure(a->loadRemoteConnInfo(b_name, conn_b) == NIXL_SUCCESS,
                         "loadRemoteConnInfo(A, B) failed");
    nixl_exit_on_failure(b->loadRemoteConnInfo(a_name, conn_a) == NIXL_SUCCESS,
                         "loadRemoteConnInfo(B, A) failed");
    nixl_exit_on_failure(a->connect(b_name) == NIXL_SUCCESS, "connect A->B failed");
    nixl_exit_on_failure(b->connect(a_name) == NIXL_SUCCESS, "connect B->A failed");
    return conn_a;
}

// WRITEs without a notification go without remote CQ data, so the target tracks nothing for
// them; before, each left an entry behind that nothing ever removed. Only a provider whose
// notified writes carry remote CQ data can show the difference.
static void
runUntrackedWriteScenario(const string &a_name, const string &b_name) {
    auto *a = createEngine(a_name);
    auto *b = createEngine(b_name);
    connectPair(a, a_name, b, b_name);
    if (TestAccess::notifiesAfterCompletion(*b)) {
        cout << "libfabric_pending_notif_test: writes carry no remote CQ data on this provider, "
                "so the untracked-WRITE check cannot fail"
             << endl;
    }

    vector<char> src(kLen, 'u'), dst(kLen, 0);
    nixlBackendMD *src_md = registerBuf(a, src.data());
    nixlBackendMD *dst_md = registerBuf(b, dst.data());
    nixlBlobDesc remote;
    remote.addr = reinterpret_cast<uintptr_t>(dst.data());
    remote.len = kLen;
    remote.devId = 0;
    nixl_exit_on_failure(b->getPublicData(dst_md, remote.metaInfo) == NIXL_SUCCESS,
                         "getPublicData failed");
    nixlBackendMD *remote_md = nullptr;
    nixl_exit_on_failure(a->loadRemoteMD(remote, DRAM_SEG, b_name, remote_md) == NIXL_SUCCESS,
                         "loadRemoteMD failed");
    nixl_meta_dlist_t local_descs(DRAM_SEG), remote_descs(DRAM_SEG);
    nixlMetaDesc ld;
    ld.addr = reinterpret_cast<uintptr_t>(src.data());
    ld.len = kLen;
    ld.devId = 0;
    ld.metadataP = src_md;
    local_descs.addDesc(ld);
    nixlMetaDesc rd;
    rd.addr = reinterpret_cast<uintptr_t>(dst.data());
    rd.len = kLen;
    rd.devId = 0;
    rd.metadataP = remote_md;
    remote_descs.addDesc(rd);

    const optional<uint32_t> a_idx = TestAccess::agentIndex(*b, a_name);
    nixl_exit_on_failure(a_idx.has_value(), "target has no index for the initiator");
    for (int k = 0; k < kIters; ++k) {
        nixlBackendReqH *handle = nullptr;
        nixl_exit_on_failure(
            a->prepXfer(NIXL_WRITE, local_descs, remote_descs, b_name, handle, nullptr) ==
                NIXL_SUCCESS,
            "prepXfer failed");
        nixl_status_t st = a->postXfer(NIXL_WRITE, local_descs, remote_descs, b_name, handle);
        nixl_exit_on_failure(st >= NIXL_SUCCESS, "postXfer failed");
        const auto deadline = steady_clock::now() + kNotifWait;
        while (st == NIXL_IN_PROG && steady_clock::now() < deadline) {
            st = a->checkXfer(handle);
        }
        nixl_exit_on_failure(st == NIXL_SUCCESS, "WRITE did not complete");
        nixl_exit_on_failure(a->releaseReqH(handle) == NIXL_SUCCESS, "releaseReqH failed");
    }
    // Let the target's progress thread reap anything the writes produced
    this_thread::sleep_for(milliseconds(200));
    notif_list_t stray;
    nixl_exit_on_failure(b->getNotifs(stray) >= NIXL_SUCCESS, "getNotifs failed");
    nixl_exit_on_failure(stray.empty(), "a WRITE without a notification produced one");
    nixl_exit_on_failure(dst[0] == 'u' && dst[kLen - 1] == 'u', "the data did not arrive");
    const size_t pending = TestAccess::pendingCount(*b, *a_idx);
    nixl_exit_on_failure(pending == 0,
                         to_string(pending) +
                             " pending notification(s) left by WRITEs without a notification");
    cout << "libfabric_pending_notif_test OK (untracked WRITEs): " << kIters
         << " WRITEs without a notification left the target nothing to track" << endl;

    a->unloadMD(remote_md);
    a->deregisterMem(src_md);
    b->deregisterMem(dst_md);
    a->disconnect(b_name);
    b->disconnect(a_name);
    delete b;
    delete a;
}

static nixlLibfabricEngine *
createEngineWithTimeout(const string &name, const char *timeout, bool progress_thread = true) {
    if (timeout) {
        nixl_exit_on_failure(setenv("NIXL_LIBFABRIC_PENDING_NOTIF_TIMEOUT", timeout, 1) == 0,
                             "setenv failed");
    }
    auto *engine = createEngine(name, progress_thread);
    nixl_exit_on_failure(unsetenv("NIXL_LIBFABRIC_PENDING_NOTIF_TIMEOUT") == 0, "unsetenv failed");
    return engine;
}

// Which stale entries are dropped. A live sender's entry whose notification is incomplete (writes
// only, or some fragments) is dropped after the notification age; one with its notification waits
// for its writes until the longer writes age. A disconnected sender's entries survive the
// disconnect, its source addresses stay mapped so late traffic is still attributed, and they are
// dropped after the notification age. A reconnect takes the addresses over but not the entries.
static void
runStaleNotificationScenario(const string &a_name, const string &b_name) {
    auto *a = createEngineWithTimeout(a_name, nullptr);
    auto *b = createEngineWithTimeout(b_name, nullptr);
    const seconds age = TestAccess::staleAge(*b);
    const seconds writes_age = TestAccess::writesAge(*b);
    nixl_exit_on_failure(writes_age > age, "the writes age is not longer than the notif age");

    const string conn_a = connectPair(a, a_name, b, b_name);
    const optional<uint32_t> a_idx = TestAccess::agentIndex(*b, a_name);
    nixl_exit_on_failure(a_idx.has_value(), "target has no index for the initiator");
    const fi_addr_t a_addr = TestAccess::firstRemoteAddr(*b, a_name);
    nixl_exit_on_failure(TestAccess::isLive(*b, *a_idx), "a connected sender is not live");

    // 1 s margins keep the checks clear of the time the calls take
    const auto before = steady_clock::now();
    b->addReceivedXferId(123, *a_idx); // writes, no notification
    TestAccess::addNotification(*b, *a_idx, 125, 0, a_name, "part", 3); // 1 of 3 fragments
    TestAccess::addNotification(*b, *a_idx, 124, 5, a_name, "slow"); // waits for its writes
    const auto after = steady_clock::now();
    nixl_exit_on_failure(TestAccess::reapStale(*b, before + age - seconds(1)) == 0,
                         "an entry younger than the stale age was dropped");
    nixl_exit_on_failure(TestAccess::reapStale(*b, after + age + seconds(1)) == 2 &&
                             TestAccess::receivedWrites(*b, *a_idx, 123) == -1 &&
                             TestAccess::receivedWrites(*b, *a_idx, 125) == -1,
                         "a live sender's incomplete notifications were not dropped once stale");
    nixl_exit_on_failure(TestAccess::receivedWrites(*b, *a_idx, 124) == 0,
                         "a live sender's notification waiting for its writes was dropped early");
    nixl_exit_on_failure(TestAccess::reapStale(*b, after + writes_age + seconds(1)) == 1 &&
                             TestAccess::pendingCount(*b, *a_idx) == 0,
                         "a notification waiting for its writes was kept past the writes age");

    // The disconnect keeps both a new entry and the source mapping
    TestAccess::addNotification(*b, *a_idx, 126, 5, a_name, "late");
    const auto disconnected = steady_clock::now();
    b->disconnect(a_name);
    nixl_exit_on_failure(TestAccess::pendingCount(*b, *a_idx) == 1,
                         "disconnect dropped a notification that may still complete");
    nixl_exit_on_failure(TestAccess::resolveOnRail0(*b, a_addr) == *a_idx,
                         "disconnect unmapped the peer's source address");
    nixl_exit_on_failure(!TestAccess::isLive(*b, *a_idx), "the disconnected index is still live");
    nixl_exit_on_failure(TestAccess::reapStale(*b, disconnected + age + seconds(1)) == 1 &&
                             TestAccess::pendingCount(*b, *a_idx) == 0,
                         "a disconnected sender's notification was not dropped after the "
                         "notification age");

    // Long after the disconnect, a late complete notification from the peer still ages out with
    // the notification age: the index stays not live however much time passes
    TestAccess::reapStale(*b, disconnected + 2 * writes_age);
    TestAccess::addNotification(*b, *a_idx, 127, 5, a_name, "later");
    const auto later = steady_clock::now();
    nixl_exit_on_failure(TestAccess::reapStale(*b, later + age + seconds(1)) == 1 &&
                             TestAccess::pendingCount(*b, *a_idx) == 0,
                         "a late notification from a long-disconnected sender waited for the "
                         "writes age");

    // A reconnect takes over the address (if the provider hands it back) but not the old entries
    b->addReceivedXferId(200, *a_idx);
    nixl_exit_on_failure(b->loadRemoteConnInfo(a_name, conn_a) == NIXL_SUCCESS,
                         "reconnect loadRemoteConnInfo(B, A) failed");
    nixl_exit_on_failure(b->connect(a_name) == NIXL_SUCCESS, "reconnect B->A failed");
    const optional<uint32_t> a_idx2 = TestAccess::agentIndex(*b, a_name);
    nixl_exit_on_failure(a_idx2.has_value() && *a_idx2 != *a_idx,
                         "the reconnect did not get a new index");
    if (TestAccess::firstRemoteAddr(*b, a_name) == a_addr) {
        nixl_exit_on_failure(TestAccess::resolveOnRail0(*b, a_addr) == *a_idx2,
                             "the reconnect did not take over the old index's address");
    } else {
        cout << "libfabric_pending_notif_test: this provider gives a reconnect new fi_addrs"
             << endl;
    }
    nixl_exit_on_failure(TestAccess::receivedWrites(*b, *a_idx, 200) == 1 &&
                             TestAccess::receivedWrites(*b, *a_idx2, 200) == -1,
                         "the reconnect moved the old index's entries");

    cout << "libfabric_pending_notif_test OK (stale notifications): incomplete ones dropped after "
            "the notification age, complete ones after the writes age, a disconnected sender's "
            "after the notification age"
         << endl;

    b->disconnect(a_name);
    a->disconnect(b_name);
    // A is the holder replying to B: its CM thread must be gone before B frees its buffers
    delete a;
    delete b;
}

// Sweeps run after a progress pass that found the CQs empty, and after four intervals regardless,
// so steady traffic cannot starve them. A 1 s stale age makes the interval 1 s. No progress thread,
// which would sweep by itself.
static void
runSweepTimingScenario() {
    auto *g = createEngineWithTimeout("AgentH", "1", false);
    TestAccess::maybeReap(*g, true); // the first sweep, now
    g->addReceivedXferId(7, 4242);
    this_thread::sleep_for(milliseconds(1500));
    TestAccess::maybeReap(*g, false);
    nixl_exit_on_failure(TestAccess::pendingCount(*g, 4242) == 1,
                         "a sweep ran while the CQs were busy, before four intervals");
    TestAccess::maybeReap(*g, true);
    nixl_exit_on_failure(TestAccess::pendingCount(*g, 4242) == 0,
                         "a drained pass after the interval did not sweep");
    g->addReceivedXferId(8, 4242);
    this_thread::sleep_for(milliseconds(4500));
    TestAccess::maybeReap(*g, false);
    nixl_exit_on_failure(TestAccess::pendingCount(*g, 4242) == 0,
                         "busy CQs starved the sweep past four intervals");
    cout << "libfabric_pending_notif_test OK (sweep timing): after a drained pass, or after four "
            "intervals regardless"
         << endl;
    delete g;
}

// The ages come from NIXL_LIBFABRIC_PENDING_{NOTIF,WRITES}_TIMEOUT: 0 keeps every entry, and a
// negative, malformed or too large value leaves the default
static void
runTimeoutConfigScenario() {
    struct {
        const char *value;
        seconds expected;
    } const cases[] = {{"0", seconds(0)},
                       {"30", seconds(30)},
                       {"-1", nixlLibfabricEngine::kStalePendingAge},
                       {"99999999999", nixlLibfabricEngine::kStalePendingAge},
                       {"5s", nixlLibfabricEngine::kStalePendingAge}};

    for (const auto &c : cases) {
        auto *g = createEngineWithTimeout("AgentG", c.value);
        nixl_exit_on_failure(TestAccess::staleAge(*g) == c.expected,
                             string("NIXL_LIBFABRIC_PENDING_NOTIF_TIMEOUT=") + c.value + " gave " +
                                 to_string(TestAccess::staleAge(*g).count()) + " s");
        if (c.expected.count() == 0) {
            g->addReceivedXferId(7, 4242);
            nixl_exit_on_failure(
                TestAccess::reapStale(*g, steady_clock::now() + hours(24)) == 0 &&
                    TestAccess::pendingCount(*g, 4242) == 1,
                "an entry was dropped with NIXL_LIBFABRIC_PENDING_NOTIF_TIMEOUT=0");
        }
        delete g;
    }
    nixl_exit_on_failure(setenv("NIXL_LIBFABRIC_PENDING_WRITES_TIMEOUT", "1200", 1) == 0,
                         "setenv failed");
    auto *w = createEngineWithTimeout("AgentW", nullptr);
    nixl_exit_on_failure(unsetenv("NIXL_LIBFABRIC_PENDING_WRITES_TIMEOUT") == 0, "unsetenv failed");
    nixl_exit_on_failure(TestAccess::writesAge(*w) == seconds(1200),
                         "NIXL_LIBFABRIC_PENDING_WRITES_TIMEOUT=1200 was not applied");
    delete w;
    cout << "libfabric_pending_notif_test OK (timeouts): 0 never drops, invalid values keep the "
            "default, the writes age is set separately"
         << endl;
}

int
main() {
    // Each scenario sets the stale age it relies on
    nixl_exit_on_failure(unsetenv("NIXL_LIBFABRIC_PENDING_NOTIF_TIMEOUT") == 0, "unsetenv failed");
    runUntrackedWriteScenario("AgentA", "AgentB");
    nixl_exit_on_failure(unsetenv("NIXL_LIBFABRIC_PENDING_WRITES_TIMEOUT") == 0, "unsetenv failed");
    runStaleNotificationScenario("AgentE", "AgentF");
    runSweepTimingScenario();
    runTimeoutConfigScenario();
    return 0;
}
