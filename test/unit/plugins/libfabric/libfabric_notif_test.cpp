/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// A notified WRITE's notification, with the progress thread, on whatever provider this host has:
// it arrives without the initiator polling checkXfer(), the data is in place when it does, and it
// arrives exactly once. On a notify-after-completion provider the initiator sends it from the
// progress thread once the writes complete; on the others it goes out at postXfer().
//
// A notified READ's notification is sent once the data has arrived, from the progress thread:
// checkXfer() reports the READ complete only once that notification is on the wire, so a
// genNotif() sent after it reaches the target second.
//
// Runs once with the provider's own notification mode and once with
// NIXL_LIBFABRIC_NOTIFY_AFTER_COMPLETION=1, which forces notify-after-completion on any provider.

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "libfabric_backend.h"
#include "test_utils.h"

using namespace std;
using namespace std::chrono;

static const size_t kLen = 1 << 20;
static const int kIters = 50;
static const seconds kNotifWait(10);
static const int kProgressThreadDelayUs = 100;

static nixlLibfabricEngine *
createEngine(const string &name) {
    nixlBackendInitParams init;
    nixl_b_params_t custom_params;

    init.enableProgTh = true;
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

static void
runScenario(const string &mode, const string &a_name, const string &b_name) {
    auto *a = createEngine(a_name);
    auto *b = createEngine(b_name);

    string conn_a, conn_b;
    nixl_exit_on_failure(a->getConnInfo(conn_a) == NIXL_SUCCESS, "getConnInfo(A) failed");
    nixl_exit_on_failure(b->getConnInfo(conn_b) == NIXL_SUCCESS, "getConnInfo(B) failed");
    nixl_exit_on_failure(a->loadRemoteConnInfo(b_name, conn_b) == NIXL_SUCCESS,
                         "loadRemoteConnInfo(A, B) failed");
    nixl_exit_on_failure(b->loadRemoteConnInfo(a_name, conn_a) == NIXL_SUCCESS,
                         "loadRemoteConnInfo(B, A) failed");
    nixl_exit_on_failure(a->connect(b_name) == NIXL_SUCCESS, "connect A->B failed");
    nixl_exit_on_failure(b->connect(a_name) == NIXL_SUCCESS, "connect B->A failed");

    vector<char> src(kLen), dst(kLen, 0);
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

    double max_ms = 0;
    for (int k = 0; k < kIters; ++k) {
        const char pattern = static_cast<char>('a' + k % 26);
        memset(src.data(), pattern, kLen);

        nixl_opt_b_args_t opt_args;
        opt_args.notifMsg = "notif-" + to_string(k);
        opt_args.hasNotif = true;
        nixlBackendReqH *handle = nullptr;
        nixl_exit_on_failure(
            a->prepXfer(NIXL_WRITE, local_descs, remote_descs, b_name, handle, &opt_args) ==
                NIXL_SUCCESS,
            "prepXfer failed");
        const auto t0 = steady_clock::now();
        nixl_status_t st =
            a->postXfer(NIXL_WRITE, local_descs, remote_descs, b_name, handle, &opt_args);
        nixl_exit_on_failure(st >= NIXL_SUCCESS, "postXfer failed");

        // The initiator does not poll: the notification must arrive on its own.
        const notif_list_t notifs = waitNotifs(b);
        max_ms = max(max_ms, duration<double, milli>(steady_clock::now() - t0).count());
        nixl_exit_on_failure(!notifs.empty(),
                             "iteration " + to_string(k) + ": no notification within " +
                                 to_string(kNotifWait.count()) +
                                 " s while the initiator did not poll checkXfer()");
        nixl_exit_on_failure(notifs.size() == 1,
                             "iteration " + to_string(k) + ": " + to_string(notifs.size()) +
                                 " notifications for one transfer");
        nixl_exit_on_failure(notifs.front().first == a_name &&
                                 notifs.front().second == opt_args.notifMsg,
                             "iteration " + to_string(k) + ": wrong notification '" +
                                 notifs.front().second + "' from " + notifs.front().first);
        for (size_t off = 0; off < kLen; off += 4096) {
            nixl_exit_on_failure(dst[off] == pattern && dst[min(off + 4095, kLen - 1)] == pattern,
                                 "iteration " + to_string(k) +
                                     ": notification arrived before the data at offset " +
                                     to_string(off));
        }

        // Now the initiator's own view, which must not send the notification again.
        const auto deadline = steady_clock::now() + kNotifWait;
        while (st == NIXL_IN_PROG && steady_clock::now() < deadline) {
            st = a->checkXfer(handle);
        }
        nixl_exit_on_failure(st == NIXL_SUCCESS,
                             "iteration " + to_string(k) + ": transfer did not complete");
        nixl_exit_on_failure(a->releaseReqH(handle) == NIXL_SUCCESS, "releaseReqH failed");
    }

    // A duplicate would arrive shortly after its original.
    this_thread::sleep_for(milliseconds(200));
    notif_list_t extra;
    nixl_exit_on_failure(b->getNotifs(extra) >= NIXL_SUCCESS, "getNotifs failed");
    nixl_exit_on_failure(extra.empty(),
                         to_string(extra.size()) + " duplicate notification(s) arrived");

    for (int k = 0; k < kIters; ++k) {
        nixl_opt_b_args_t opt_args;
        opt_args.notifMsg = "read-notif-" + to_string(k);
        opt_args.hasNotif = true;
        nixlBackendReqH *handle = nullptr;
        nixl_exit_on_failure(
            a->prepXfer(NIXL_READ, local_descs, remote_descs, b_name, handle, &opt_args) ==
                NIXL_SUCCESS,
            "prepXfer(READ) failed");
        nixl_status_t st =
            a->postXfer(NIXL_READ, local_descs, remote_descs, b_name, handle, &opt_args);
        nixl_exit_on_failure(st >= NIXL_SUCCESS, "postXfer(READ) failed");
        const auto deadline = steady_clock::now() + kNotifWait;
        while (st == NIXL_IN_PROG && steady_clock::now() < deadline) {
            st = a->checkXfer(handle);
        }
        nixl_exit_on_failure(st == NIXL_SUCCESS,
                             "READ iteration " + to_string(k) + ": transfer did not complete");
        nixl_exit_on_failure(a->releaseReqH(handle) == NIXL_SUCCESS, "releaseReqH failed");

        const string after = "after-read-" + to_string(k);
        nixl_exit_on_failure(a->genNotif(b_name, after) == NIXL_SUCCESS, "genNotif failed");

        notif_list_t notifs;
        const auto notif_deadline = steady_clock::now() + kNotifWait;
        while (notifs.size() < 2 && steady_clock::now() < notif_deadline) {
            notif_list_t more = waitNotifs(b);
            notifs.insert(notifs.end(), more.begin(), more.end());
        }
        nixl_exit_on_failure(notifs.size() == 2,
                             "READ iteration " + to_string(k) + ": " + to_string(notifs.size()) +
                                 " notifications instead of the READ's and genNotif()'s");
        nixl_exit_on_failure(notifs[0].second == opt_args.notifMsg && notifs[1].second == after,
                             "READ iteration " + to_string(k) + ": got '" + notifs[0].second +
                                 "' then '" + notifs[1].second +
                                 "': the READ's notification must precede a later genNotif()");
    }

    cout << "libfabric_notif_test OK (" << mode << "): " << kIters
         << " notified WRITEs, each notification delivered once without the initiator polling "
            "(slowest "
         << max_ms << " ms), and " << kIters
         << " notified READs whose notification preceded a later genNotif()" << endl;

    a->unloadMD(remote_md);
    a->deregisterMem(src_md);
    b->deregisterMem(dst_md);
    a->disconnect(b_name);
    b->disconnect(a_name);
    delete b;
    delete a;
}

int
main() {
    runScenario("provider default", "AgentA", "AgentB");
    // Read when the engines create their rails.
    nixl_exit_on_failure(setenv("NIXL_LIBFABRIC_NOTIFY_AFTER_COMPLETION", "1", 1) == 0,
                         "setenv failed");
    runScenario("notify-after-completion", "AgentC", "AgentD");
    return 0;
}
