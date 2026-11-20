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

// The report a failed notified WRITE sends its target, on whatever provider this host has: it is
// queued without blocking the caller, reaches a live peer with and without a progress thread, and
// the target releases the notification as a failed transfer's; after a failed WRITE has broken
// the connection the message is resent and the notification still follows; when the initiator
// stays quiet past the target's pending_notif_timeout, the target keeps the failure it recorded
// and still releases the notification as failed;
// to a peer that is gone it is given up after the post -FI_EAGAIN timeout, with no notification.
// A reserved XFER_ID is not handed out. A report for a transfer posted on a connection that has
// since been replaced never reaches the replacement.

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
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
    // The engine's current connection to @p remote_agent
    static weak_ptr<nixlLibfabricConnection>
    connectionTo(const nixlLibfabricEngine &engine, const string &remote_agent) {
        const lock_guard<mutex> lock(engine.connection_state_mutex_);
        return engine.connections_.at(remote_agent);
    }

    // Report a failed notified WRITE (XFER_ID @p xfer_id, none of its writes done) to
    // @p remote_agent as checkXfer() does, with the notification @p msg owed. The transfer was
    // posted on @p posted_on, by default the current connection.
    static void
    reportFailure(const nixlLibfabricEngine &engine,
                  const string &remote_agent,
                  uint16_t xfer_id,
                  const string &msg,
                  weak_ptr<nixlLibfabricConnection> posted_on = {}) {
        nixlLibfabricBackendH handle(NIXL_WRITE, remote_agent);
        handle.post_connection_ =
            posted_on.expired() ? connectionTo(engine, remote_agent) : posted_on;
        handle.has_notif = true;
        handle.notif_deferred = true; // owed on every provider, as after a partial post
        handle.post_xfer_id = xfer_id;
        engine.fragmentNotificationMessage(
            msg, engine.localAgent, handle.total_notif_msg_len, handle.binary_notifs);
        engine.notifXferFailure(&handle, NIXL_ERR_BACKEND);
    }

    // Reports not finished yet; waits for a step in progress, whose reports the driver holds
    static size_t
    queuedReports(const nixlLibfabricEngine &engine) {
        const lock_guard<mutex> driving(engine.failure_reports_drive_mutex_);
        const lock_guard<mutex> lock(engine.failure_reports_mutex_);
        return engine.failure_reports_.size();
    }

    // Notifications @p engine released as a failed transfer's
    static size_t
    failedReleased(const nixlLibfabricEngine &engine) {
        return engine.failed_notifs_released_.load();
    }

    // The provider of @p engine's rail 0
    static string
    providerName(nixlLibfabricEngine &engine) {
        return engine.rail_manager_.getRail(0).provider_name;
    }

    // Notifications @p engine is still assembling or waiting for writes for
    static size_t
    pendingNotifs(nixlLibfabricEngine &engine) {
        const lock_guard<mutex> lock(engine.receiver_tracking_mutex_);
        return engine.pending_notifications_.size();
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
};

using TestAccess = nixlLibfabricEngineTestAccess;

static const int kProgressThreadDelayUs = 100;
static const size_t kLen = 1 << 20;
static const seconds kArriveWait(10);

static nixlLibfabricEngine *
createEngine(const string &name,
             bool progress_thread,
             const char *eagain_timeout = nullptr,
             const char *pending_notif_timeout = nullptr) {
    nixlBackendInitParams init;
    nixl_b_params_t custom_params;
    if (eagain_timeout) {
        custom_params["post_eagain_timeout"] = eagain_timeout;
    }
    if (pending_notif_timeout) {
        custom_params["pending_notif_timeout"] = pending_notif_timeout;
    }

    init.enableProgTh = progress_thread;
    init.pthrDelay = kProgressThreadDelayUs;
    init.localAgent = name;
    init.customParams = &custom_params;
    init.type = "LIBFABRIC";

    auto *engine = new nixlLibfabricEngine(&init);
    nixl_exit_on_failure(!engine->getInitErr(), "Failed to initialize engine " + name);
    return engine;
}

static void
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

// A notified WRITE to a buffer the target has deregistered fails at the target NIC. Under
// verbs;ofi_rxm that puts the RC QP into the error state, which flushes the transfer-error message
// sent right after on the same connection; it must be resent once the provider reconnects, and
// the notification must follow it. The target then releases the notification with its "data
// incomplete" warning. Without the resend both were flushed and the target saw nothing.
// checkXfer() reports the failure without waiting for any of this, and the handle is released
// before the report has gone out.
static void
runBrokenConnectionScenario() {
    auto *a = createEngine("AgentG", true, "10");
    auto *b = createEngine("AgentH", true);
    connectPair(a, "AgentG", b, "AgentH");

    vector<char> src(kLen, 'x'), dst(kLen, 0);
    nixlBackendMD *src_md = registerBuf(a, src.data());
    nixlBackendMD *dst_md = registerBuf(b, dst.data());
    nixlBlobDesc remote;
    remote.addr = reinterpret_cast<uintptr_t>(dst.data());
    remote.len = kLen;
    remote.devId = 0;
    nixl_exit_on_failure(b->getPublicData(dst_md, remote.metaInfo) == NIXL_SUCCESS,
                         "getPublicData failed");
    nixlBackendMD *remote_md = nullptr;
    nixl_exit_on_failure(a->loadRemoteMD(remote, DRAM_SEG, "AgentH", remote_md) == NIXL_SUCCESS,
                         "loadRemoteMD failed");
    nixl_exit_on_failure(b->deregisterMem(dst_md) == NIXL_SUCCESS, "deregisterMem failed");

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

    nixl_opt_b_args_t opt;
    opt.notifMsg = "broken";
    opt.hasNotif = true;
    nixlBackendReqH *handle = nullptr;
    nixl_exit_on_failure(
        a->prepXfer(NIXL_WRITE, local_descs, remote_descs, "AgentH", handle, &opt) == NIXL_SUCCESS,
        "prepXfer failed");
    nixl_status_t st = a->postXfer(NIXL_WRITE, local_descs, remote_descs, "AgentH", handle, &opt);
    const auto deadline = steady_clock::now() + kArriveWait;
    milliseconds longest_check(0);
    while (st == NIXL_IN_PROG && steady_clock::now() < deadline) {
        const auto before = steady_clock::now();
        st = a->checkXfer(handle);
        longest_check =
            max(longest_check, duration_cast<milliseconds>(steady_clock::now() - before));
    }
    nixl_exit_on_failure(a->releaseReqH(handle) == NIXL_SUCCESS, "releaseReqH failed");
    if (st == NIXL_SUCCESS) {
        // A provider that does not check the remote key (tcp, sockets) cannot break the connection
        cout << "libfabric_xfer_error_test: a WRITE to a deregistered buffer succeeded on this "
                "provider, so the broken-connection resend is not exercised"
             << endl;
    } else {
        nixl_exit_on_failure(st < NIXL_SUCCESS, "the WRITE to a deregistered buffer hung");
        nixl_exit_on_failure(longest_check < milliseconds(1000),
                             "a checkXfer() call took " + to_string(longest_check.count()) +
                                 " ms; it must not wait for the failure report");
        notif_list_t notifs;
        const auto notif_deadline = steady_clock::now() + kArriveWait;
        while (notifs.empty() && steady_clock::now() < notif_deadline) {
            nixl_exit_on_failure(b->getNotifs(notifs) >= NIXL_SUCCESS, "getNotifs failed");
            this_thread::sleep_for(milliseconds(1));
        }
        // On tcp and sockets a send completion only means the socket took the message: a
        // notification sent at postXfer() can complete and still be lost when the failed WRITE
        // tears the connection down, and nothing tells the initiator to send it again (a known
        // limitation). The target must still hold the failure the error message recorded.
        const string provider = TestAccess::providerName(*b);
        if (notifs.empty() && (provider == "tcp" || provider == "sockets")) {
            nixl_exit_on_failure(TestAccess::pendingNotifs(*b) == 1,
                                 "the target holds no failure record for the failed WRITE");
            cout << "libfabric_xfer_error_test OK (broken connection): WRITE failed with " << st
                 << "; on " << provider
                 << " the notification sent at postXfer() was lost with the connection (known "
                    "limitation), the target holds the failure"
                 << endl;
            a->unloadMD(remote_md);
            a->deregisterMem(src_md);
            a->disconnect("AgentH");
            b->disconnect("AgentG");
            delete b;
            delete a;
            return;
        }
        nixl_exit_on_failure(notifs.size() == 1 && notifs.front().second == "broken",
                             "the target did not get the failed WRITE's notification after its "
                             "transfer-error message");
        nixl_exit_on_failure(TestAccess::failedReleased(*b) == 1,
                             "the target released the failed WRITE's notification as a "
                             "successful transfer's");
        cout << "libfabric_xfer_error_test OK (broken connection): WRITE failed with " << st
             << ", longest checkXfer() " << longest_check.count()
             << " ms, the target still got the error and the notification" << endl;
    }
    a->unloadMD(remote_md);
    a->deregisterMem(src_md);
    a->disconnect("AgentH");
    b->disconnect("AgentG");
    delete b;
    delete a;
}

// Polls the target's notifications until one arrives or kArriveWait passes. With @p drive, also
// calls the initiator's getNotifs(), which steps its reports when it has no progress thread.
static notif_list_t
waitNotifs(nixlLibfabricEngine *target, nixlLibfabricEngine *drive = nullptr) {
    notif_list_t notifs, ignored;
    const auto deadline = steady_clock::now() + kArriveWait;
    while (notifs.empty() && steady_clock::now() < deadline) {
        nixl_exit_on_failure(target->getNotifs(notifs) >= NIXL_SUCCESS, "getNotifs failed");
        if (drive) {
            nixl_exit_on_failure(drive->getNotifs(ignored) >= NIXL_SUCCESS, "getNotifs failed");
        }
        this_thread::sleep_for(milliseconds(1));
    }
    return notifs;
}

// A live peer gets the transfer-error message and then the notification, which it releases. The
// target always runs a progress thread so a lazily connecting provider accepts the initiator; the
// initiator's may be off, in which case its getNotifs() calls step the report.
static void
runDeliveredScenario(bool initiator_progress_thread) {
    const string a_name = initiator_progress_thread ? "AgentA" : "AgentC";
    const string b_name = initiator_progress_thread ? "AgentB" : "AgentD";
    auto *a = createEngine(a_name, initiator_progress_thread);
    auto *b = createEngine(b_name, true);
    connectPair(a, a_name, b, b_name);

    const auto start = steady_clock::now();
    TestAccess::reportFailure(*a, b_name, 77, "delivered");
    const auto took = duration_cast<milliseconds>(steady_clock::now() - start);
    const notif_list_t notifs = waitNotifs(b, initiator_progress_thread ? nullptr : a);
    nixl_exit_on_failure(notifs.size() == 1 && notifs.front().first == a_name &&
                             notifs.front().second == "delivered",
                         "the target did not get the failed transfer's notification");
    // The notification counts no writes, so only this tells it from a successful transfer's
    nixl_exit_on_failure(TestAccess::failedReleased(*b) == 1,
                         "the target released the failed transfer's notification as a successful "
                         "transfer's");
    const auto deadline = steady_clock::now() + kArriveWait;
    while (TestAccess::queuedReports(*a) != 0 && steady_clock::now() < deadline) {
        notif_list_t ignored;
        nixl_exit_on_failure(a->getNotifs(ignored) >= NIXL_SUCCESS, "getNotifs failed");
        this_thread::sleep_for(milliseconds(1));
    }
    nixl_exit_on_failure(TestAccess::queuedReports(*a) == 0, "a delivered report stayed queued");
    cout << "libfabric_xfer_error_test OK (delivered, initiator progress thread "
         << (initiator_progress_thread ? "on" : "off") << "): queued in " << took.count()
         << " ms, the error and then the notification arrived" << endl;

    a->disconnect(b_name);
    b->disconnect(a_name);
    delete b;
    delete a;
}

// The initiator, without a progress thread, does not poll for longer than the target's
// pending_notif_timeout (opt-in, here 1 s). The target must keep the failure the delivered error
// message recorded, since its sender is connected, and release the notification that follows as a
// failed transfer's.
static void
runQuietInitiatorScenario() {
    constexpr int kPendingTimeoutS = 1;
    auto *a = createEngine("AgentI", false);
    auto *b = createEngine("AgentJ", true, nullptr, to_string(kPendingTimeoutS).c_str());
    connectPair(a, "AgentI", b, "AgentJ");

    TestAccess::reportFailure(*a, "AgentJ", 79, "quiet");
    // Long enough for the error to arrive and for a dropped entry to have been swept
    const auto quiet_until = steady_clock::now() + seconds(3 * kPendingTimeoutS + 1);
    bool marker_seen = false;
    while (steady_clock::now() < quiet_until) {
        marker_seen = marker_seen || TestAccess::pendingNotifs(*b) != 0;
        this_thread::sleep_for(milliseconds(10));
    }
    nixl_exit_on_failure(!marker_seen || TestAccess::pendingNotifs(*b) != 0,
                         "the target dropped a connected sender's failure record");

    const notif_list_t notifs = waitNotifs(b, a);
    nixl_exit_on_failure(notifs.size() == 1 && notifs.front().second == "quiet",
                         "the target did not get the failed transfer's notification");
    nixl_exit_on_failure(TestAccess::failedReleased(*b) == 1,
                         "after a quiet initiator, the target released the failed transfer's "
                         "notification as a successful transfer's");
    cout << "libfabric_xfer_error_test OK (quiet initiator): "
         << (marker_seen ? "the target kept the failure record past pending_notif_timeout and "
                           "released the notification as failed" :
                           "the error did not arrive before the initiator polled on this provider, "
                           "so the kept record is not exercised")
         << endl;

    a->disconnect("AgentJ");
    b->disconnect("AgentI");
    delete b;
    delete a;
}

// getNextXferId() skips an XFER_ID a queued failure report holds, until it is released
static void
runReservedXferIdCheck() {
    const uint16_t id = LibfabricUtils::getNextXferId();
    const uint16_t held = static_cast<uint16_t>(id + 1) == 0 ? 1 : id + 1;
    LibfabricUtils::reserveXferId(held);
    const uint16_t next = LibfabricUtils::getNextXferId();
    LibfabricUtils::releaseXferId(held);
    nixl_exit_on_failure(next != held, "getNextXferId() handed out a reserved XFER_ID");
    // Released: handed out again once the counter comes round
    uint16_t again = 0;
    for (int i = 0; i < 65536 && again != held; ++i) {
        again = LibfabricUtils::getNextXferId();
    }
    nixl_exit_on_failure(again == held, "a released XFER_ID was not handed out again");
    cout << "libfabric_xfer_error_test OK (reserved XFER_ID skipped)" << endl;
}

// A peer that is gone: every attempt fails (in its CQ, or at post after -FI_EAGAIN), the message
// is resent, and once the post -FI_EAGAIN timeout has passed the report is given up. Queuing it
// does not wait.
static void
runGonePeerScenario() {
    constexpr int kTimeoutS = 2;
    auto *a = createEngine("AgentE", true, to_string(kTimeoutS).c_str());
    auto *b = createEngine("AgentF", true);
    connectPair(a, "AgentE", b, "AgentF");
    delete b; // no disconnect: AgentE still holds its addresses

    const auto start = steady_clock::now();
    TestAccess::reportFailure(*a, "AgentF", 78, "gone");
    const auto queued = duration_cast<milliseconds>(steady_clock::now() - start);
    nixl_exit_on_failure(queued < milliseconds(1000),
                         "queuing a failure report took " + to_string(queued.count()) + " ms");
    // One post may itself spend the -FI_EAGAIN timeout after the resend deadline
    const auto deadline = start + seconds(2 * kTimeoutS + 5);
    while (TestAccess::queuedReports(*a) != 0 && steady_clock::now() < deadline) {
        this_thread::sleep_for(milliseconds(10));
    }
    const auto took = duration_cast<milliseconds>(steady_clock::now() - start);
    nixl_exit_on_failure(TestAccess::queuedReports(*a) == 0,
                         "a report to a peer that is gone was not given up after " +
                             to_string(took.count()) + " ms");
    cout << "libfabric_xfer_error_test OK (peer gone): queued in " << queued.count()
         << " ms, given up after " << took.count() << " ms" << endl;

    a->disconnect("AgentF");
    delete a;
}

// A transfer posted on a connection that has since been replaced (disconnect, then a new
// connection to the same agent) fails: its report must not reach the replacement. The old
// connection is no longer registered, so the report is dropped and the target gets nothing; a
// report for a transfer posted on the new connection still arrives.
static void
runReplacedConnectionScenario() {
    auto *a = createEngine("AgentK", true);
    auto *b = createEngine("AgentL", true);
    connectPair(a, "AgentK", b, "AgentL");
    // Held, so the old connection object outlives the disconnect and the report can name it
    const shared_ptr<nixlLibfabricConnection> old_conn =
        TestAccess::connectionTo(*a, "AgentL").lock();

    a->disconnect("AgentL");
    string conn_b;
    nixl_exit_on_failure(b->getConnInfo(conn_b) == NIXL_SUCCESS, "getConnInfo(B) failed");
    nixl_exit_on_failure(a->loadRemoteConnInfo("AgentL", conn_b) == NIXL_SUCCESS &&
                             a->connect("AgentL") == NIXL_SUCCESS,
                         "reconnecting A to B failed");
    nixl_exit_on_failure(TestAccess::connectionTo(*a, "AgentL").lock() != old_conn,
                         "the reconnect kept the old connection object");

    TestAccess::reportFailure(*a, "AgentL", 82, "stale", old_conn);
    const auto until = steady_clock::now() + seconds(1);
    size_t stale = 0;
    while (steady_clock::now() < until) {
        notif_list_t got;
        nixl_exit_on_failure(b->getNotifs(got) >= NIXL_SUCCESS, "getNotifs failed");
        stale += got.size();
        this_thread::sleep_for(milliseconds(5));
    }
    nixl_exit_on_failure(stale == 0,
                         "a report for a transfer posted on the replaced connection reached the "
                         "new one");
    nixl_exit_on_failure(TestAccess::queuedReports(*a) == 0,
                         "the report for the replaced connection was not dropped");

    TestAccess::reportFailure(*a, "AgentL", 83, "current");
    const notif_list_t notifs = waitNotifs(b);
    nixl_exit_on_failure(notifs.size() == 1 && notifs.front().second == "current",
                         "a report for a transfer posted on the new connection did not arrive");
    cout << "libfabric_xfer_error_test OK (replaced connection): the stale report was dropped, the "
            "current one delivered"
         << endl;

    a->disconnect("AgentL");
    b->disconnect("AgentK");
    delete b;
    delete a;
}

int
main() {
    // libfabric's MR cache would keep the deregistered buffer's registration alive, and the WRITE
    // meant to break the connection would succeed
    nixl_exit_on_failure(setenv("FI_MR_CACHE_MAX_COUNT", "0", 1) == 0, "setenv failed");
    runDeliveredScenario(true);
    runDeliveredScenario(false);
    runBrokenConnectionScenario();
    runQuietInitiatorScenario();
    runGonePeerScenario();
    runReservedXferIdCheck();
    runReplacedConnectionScenario();
    return 0;
}
