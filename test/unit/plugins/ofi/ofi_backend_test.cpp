/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Intel Corporation. All rights reserved.
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

#include <gtest/gtest.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "ofi_backend.h"
#include "common/nixl_log.h"

// The engine takes its provider from FI_PROVIDER, and libfabric reads FI_PROVIDER once per
// process, so each provider runs in its own test process (see meson.build). Changing it at
// run time only affects the engine's own provider check, which InvalidProvider relies on.
class ScopedProvider {
public:
    explicit ScopedProvider(const char *provider) {
        const char *old = getenv("FI_PROVIDER");
        if (old) {
            saved_ = old;
            hadOld_ = true;
        }
        setenv("FI_PROVIDER", provider, 1);
    }

    ~ScopedProvider() {
        if (hadOld_) {
            setenv("FI_PROVIDER", saved_.c_str(), 1);
        } else {
            unsetenv("FI_PROVIDER");
        }
    }

private:
    std::string saved_;
    bool hadOld_ = false;
};

class OfiBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        init_params.localAgent = "test_agent";
        custom_params["eq_timeout_ms"] = "100";
        init_params.customParams = &custom_params;
    }

    nixlBackendInitParams init_params;
    nixl_b_params_t custom_params;
};

TEST_F(OfiBackendTest, ConstructorBasic) {
    std::unique_ptr<nixlOfiEngine> engine(new nixlOfiEngine(&init_params));
    EXPECT_NE(engine.get(), nullptr);
}

TEST_F(OfiBackendTest, SupportMethods) {
    std::unique_ptr<nixlOfiEngine> engine(new nixlOfiEngine(&init_params));

    // true although notifications are not implemented (see README.md)
    EXPECT_TRUE(engine->supportsNotif());
    EXPECT_TRUE(engine->supportsRemote());
    EXPECT_FALSE(engine->supportsLocal());
}

TEST_F(OfiBackendTest, GetSupportedMems) {
    std::unique_ptr<nixlOfiEngine> engine(new nixlOfiEngine(&init_params));

    nixl_mem_list_t mems = engine->getSupportedMems();
    EXPECT_FALSE(mems.empty());
    EXPECT_EQ(mems[0], DRAM_SEG);
}

TEST_F(OfiBackendTest, InvalidProvider) {
    ScopedProvider invalid("invalid_provider");
    std::unique_ptr<nixlOfiEngine> engine(new nixlOfiEngine(&init_params));
    EXPECT_TRUE(engine->getInitErr());
}

TEST_F(OfiBackendTest, CustomTimeout) {
    custom_params["eq_timeout_ms"] = "500";
    std::unique_ptr<nixlOfiEngine> engine(new nixlOfiEngine(&init_params));
    EXPECT_NE(engine.get(), nullptr);
}

TEST_F(OfiBackendTest, InvalidTimeout) {
    custom_params["eq_timeout_ms"] = "invalid";
    std::unique_ptr<nixlOfiEngine> engine(new nixlOfiEngine(&init_params));
    EXPECT_NE(engine.get(), nullptr);
}

TEST_F(OfiBackendTest, TimeoutOutOfRange) {
    custom_params["eq_timeout_ms"] = "70000";
    std::unique_ptr<nixlOfiEngine> engine(new nixlOfiEngine(&init_params));
    EXPECT_NE(engine.get(), nullptr);
}

TEST_F(OfiBackendTest, ConnectWithoutRemoteInfo) {
    std::unique_ptr<nixlOfiEngine> engine(new nixlOfiEngine(&init_params));
    EXPECT_EQ(engine->connect("nonexistent_agent"), NIXL_ERR_NOT_FOUND);
}

TEST_F(OfiBackendTest, DisconnectNonexistentAgent) {
    std::unique_ptr<nixlOfiEngine> engine(new nixlOfiEngine(&init_params));
    EXPECT_EQ(engine->disconnect("nonexistent_agent"), NIXL_ERR_NOT_FOUND);
}

TEST_F(OfiBackendTest, DeregisterNullMetadata) {
    std::unique_ptr<nixlOfiEngine> engine(new nixlOfiEngine(&init_params));
    EXPECT_EQ(engine->deregisterMem(nullptr), NIXL_ERR_INVALID_PARAM);
}

TEST_F(OfiBackendTest, CheckXferWithNullHandle) {
    std::unique_ptr<nixlOfiEngine> engine(new nixlOfiEngine(&init_params));
    EXPECT_EQ(engine->checkXfer(nullptr), NIXL_ERR_INVALID_PARAM);
}

TEST_F(OfiBackendTest, ReleaseReqHWithNullHandle) {
    std::unique_ptr<nixlOfiEngine> engine(new nixlOfiEngine(&init_params));
    EXPECT_EQ(engine->releaseReqH(nullptr), NIXL_ERR_INVALID_PARAM);
}

TEST(OfiRequestTest, DefaultConstructor) {
    nixlOfiRequest request;
    EXPECT_EQ(request.cq, nullptr);
    EXPECT_EQ(request.pending(), 0u);
}

// Two engines in one process: "target" owns the remote buffer, "initiator" posts the
// transfers, on host memory over the process's FI_PROVIDER.
class OfiTransferTest : public ::testing::Test {
protected:
    static constexpr size_t kBlocks = 64; // more completions than one CQ read returns
    static constexpr size_t kBlockLen = 64 * 1024; // above any inject size

    void SetUp() override {
        target_ = makeEngine("target");
        initiator_ = makeEngine("initiator");
        if (target_->getInitErr() || initiator_->getInitErr()) {
            GTEST_SKIP() << "provider " << provider() << " not available";
        }

        std::string conn_info;
        ASSERT_EQ(target_->getConnInfo(conn_info), NIXL_SUCCESS);
        ASSERT_EQ(initiator_->loadRemoteConnInfo("target", conn_info), NIXL_SUCCESS);

        remote_.assign(kBlocks * kBlockLen, 0);
        local_.assign(kBlocks * kBlockLen, 0);
        remoteMd_ = registerDram(*target_, remote_);
        localMd_ = registerDram(*initiator_, local_);

        nixlBlobDesc blob;
        ASSERT_EQ(target_->getPublicData(remoteMd_, blob.metaInfo), NIXL_SUCCESS);
        ASSERT_EQ(initiator_->loadRemoteMD(blob, DRAM_SEG, "target", remoteKey_), NIXL_SUCCESS);
    }

    void TearDown() override {
        if (remoteKey_) {
            initiator_->unloadMD(remoteKey_);
        }
        if (localMd_) {
            initiator_->deregisterMem(localMd_);
        }
        if (remoteMd_) {
            target_->deregisterMem(remoteMd_);
        }
        initiator_.reset();
        target_.reset();
    }

    std::unique_ptr<nixlOfiEngine> makeEngine(const char *name) {
        nixlBackendInitParams params;
        params.localAgent = name;
        params.customParams = &params_;
        // As a NIXL agent does by default: software providers (tcp, shm) only serve
        // incoming RMA while the target's progress thread polls its CQ.
        params.enableProgTh = true;
        params.pthrDelay = 100;
        return std::make_unique<nixlOfiEngine>(&params);
    }

    static nixlBackendMD *registerDram(nixlOfiEngine &engine, std::vector<uint8_t> &buf) {
        nixlBlobDesc mem;
        mem.addr = reinterpret_cast<uintptr_t>(buf.data());
        mem.len = buf.size();
        mem.devId = 0;
        nixlBackendMD *md = nullptr;
        EXPECT_EQ(engine.registerMem(mem, DRAM_SEG, md), NIXL_SUCCESS);
        return md;
    }

    // One descriptor per block in [first, first + count).
    void buildLists(size_t first, size_t count, nixl_meta_dlist_t &local, nixl_meta_dlist_t &remote) {
        for (size_t b = first; b < first + count; ++b) {
            local.addDesc(nixlMetaDesc(
                reinterpret_cast<uintptr_t>(local_.data() + b * kBlockLen), kBlockLen, 0, localMd_));
            remote.addDesc(nixlMetaDesc(
                reinterpret_cast<uintptr_t>(remote_.data() + b * kBlockLen), kBlockLen, 0, remoteKey_));
        }
    }

    // Prep and post; returns the handle after checking post's status is honest.
    nixlBackendReqH *start(nixl_xfer_op_t op, size_t first, size_t count) {
        nixl_meta_dlist_t local(DRAM_SEG), remote(DRAM_SEG);
        buildLists(first, count, local, remote);
        nixlBackendReqH *handle = nullptr;
        EXPECT_EQ(initiator_->prepXfer(op, local, remote, "target", handle), NIXL_SUCCESS);
        nixl_status_t st = initiator_->postXfer(op, local, remote, "target", handle);
        EXPECT_TRUE(st == NIXL_SUCCESS || st == NIXL_IN_PROG) << "postXfer: " << st;
        // the agent never calls checkXfer after SUCCESS, so SUCCESS must mean complete
        if (st == NIXL_SUCCESS) {
            EXPECT_EQ(static_cast<nixlOfiRequest *>(handle)->pending(), 0u);
        }
        return handle;
    }

    nixl_status_t waitDone(nixlBackendReqH *handle) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        nixl_status_t st;
        while ((st = initiator_->checkXfer(handle)) == NIXL_IN_PROG &&
               std::chrono::steady_clock::now() < deadline) {
        }
        return st;
    }

    static void fill(std::vector<uint8_t> &buf, uint8_t seed) {
        for (size_t i = 0; i < buf.size(); ++i) {
            buf[i] = static_cast<uint8_t>(seed + i * 131 + (i >> 16));
        }
    }

    static std::string provider() {
        const char *p = getenv("FI_PROVIDER");
        return p ? p : "";
    }

    nixl_b_params_t params_;
    std::unique_ptr<nixlOfiEngine> target_, initiator_;
    std::vector<uint8_t> remote_, local_;
    nixlBackendMD *remoteMd_ = nullptr, *localMd_ = nullptr, *remoteKey_ = nullptr;
};

TEST_F(OfiTransferTest, ReadThenWrite) {
    fill(remote_, 1);
    nixlBackendReqH *h = start(NIXL_READ, 0, kBlocks);
    ASSERT_EQ(waitDone(h), NIXL_SUCCESS);
    EXPECT_EQ(local_, remote_);
    EXPECT_EQ(initiator_->releaseReqH(h), NIXL_SUCCESS);

    fill(local_, 7);
    h = start(NIXL_WRITE, 0, kBlocks);
    ASSERT_EQ(waitDone(h), NIXL_SUCCESS);
    EXPECT_EQ(remote_, local_);
    EXPECT_EQ(initiator_->releaseReqH(h), NIXL_SUCCESS);
}

// prepXfer used to post the transfer itself, so every transfer ran twice.
TEST_F(OfiTransferTest, PrepMovesNoData) {
    fill(remote_, 3);
    nixl_meta_dlist_t local(DRAM_SEG), remote(DRAM_SEG);
    buildLists(0, kBlocks, local, remote);
    nixlBackendReqH *h = nullptr;
    ASSERT_EQ(initiator_->prepXfer(NIXL_READ, local, remote, "target", h), NIXL_SUCCESS);

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    initiator_->checkXfer(h); // drive progress; nothing should be outstanding
    EXPECT_EQ(static_cast<nixlOfiRequest *>(h)->pending(), 0u);
    EXPECT_EQ(local_, std::vector<uint8_t>(local_.size(), 0)) << "prepXfer moved data";

    // the data posted is what the source holds at post time, not at prep time
    fill(remote_, 5);
    nixl_status_t st = initiator_->postXfer(NIXL_READ, local, remote, "target", h);
    ASSERT_TRUE(st == NIXL_SUCCESS || st == NIXL_IN_PROG);
    ASSERT_EQ(waitDone(h), NIXL_SUCCESS);
    EXPECT_EQ(local_, remote_);
    EXPECT_EQ(initiator_->releaseReqH(h), NIXL_SUCCESS);
}

// Completions share one CQ; each request must see only its own.
TEST_F(OfiTransferTest, ConcurrentRequests) {
    fill(remote_, 9);
    const size_t half = kBlocks / 2;
    nixlBackendReqH *a = start(NIXL_READ, 0, half);
    nixlBackendReqH *b = start(NIXL_READ, half, half);
    ASSERT_EQ(waitDone(b), NIXL_SUCCESS);
    ASSERT_EQ(waitDone(a), NIXL_SUCCESS);
    EXPECT_EQ(local_, remote_);
    EXPECT_EQ(initiator_->releaseReqH(a), NIXL_SUCCESS);
    EXPECT_EQ(initiator_->releaseReqH(b), NIXL_SUCCESS);
}

TEST_F(OfiTransferTest, RepostHandle) {
    nixl_meta_dlist_t local(DRAM_SEG), remote(DRAM_SEG);
    buildLists(0, kBlocks, local, remote);
    nixlBackendReqH *h = nullptr;
    ASSERT_EQ(initiator_->prepXfer(NIXL_READ, local, remote, "target", h), NIXL_SUCCESS);
    for (uint8_t round = 0; round < 3; ++round) {
        fill(remote_, 20 + round);
        nixl_status_t st = initiator_->postXfer(NIXL_READ, local, remote, "target", h);
        ASSERT_TRUE(st == NIXL_SUCCESS || st == NIXL_IN_PROG);
        if (st == NIXL_IN_PROG) {
            // reposting an active request must be refused
            nixl_status_t again = initiator_->postXfer(NIXL_READ, local, remote, "target", h);
            EXPECT_TRUE(again == NIXL_ERR_REPOST_ACTIVE ||
                        static_cast<nixlOfiRequest *>(h)->pending() == 0)
                << "repost of an active request returned " << again;
        }
        ASSERT_EQ(waitDone(h), NIXL_SUCCESS);
        EXPECT_EQ(local_, remote_) << "round " << int(round);
    }
    EXPECT_EQ(initiator_->releaseReqH(h), NIXL_SUCCESS);
}

// Releasing with operations in flight must neither crash nor leak the request
// (run under ASan/TSan to check the release/completion handoff).
TEST_F(OfiTransferTest, ReleaseInFlight) {
    for (int i = 0; i < 20; ++i) {
        nixlBackendReqH *h = start(NIXL_READ, 0, kBlocks);
        EXPECT_EQ(initiator_->releaseReqH(h), NIXL_SUCCESS);
    }
    // later transfers still complete and are credited correctly
    fill(remote_, 11);
    nixlBackendReqH *h = start(NIXL_READ, 0, kBlocks);
    ASSERT_EQ(waitDone(h), NIXL_SUCCESS);
    EXPECT_EQ(local_, remote_);
    EXPECT_EQ(initiator_->releaseReqH(h), NIXL_SUCCESS);
}

// Without SynapseAI, VRAM must be refused rather than registered as host memory.
TEST_F(OfiTransferTest, VramNeedsSynapseAI) {
    nixl_mem_list_t mems = initiator_->getSupportedMems();
    bool vram = std::find(mems.begin(), mems.end(), VRAM_SEG) != mems.end();
    if (vram) {
        GTEST_SKIP() << "SynapseAI available; VRAM is supported on " << provider();
    }
    std::vector<uint8_t> buf(4096);
    nixlBlobDesc mem;
    mem.addr = reinterpret_cast<uintptr_t>(buf.data());
    mem.len = buf.size();
    nixlBackendMD *md = nullptr;
    EXPECT_EQ(initiator_->registerMem(mem, VRAM_SEG, md), NIXL_ERR_NOT_SUPPORTED);
    EXPECT_EQ(md, nullptr);
}

int main(int argc, char **argv) {
    // must be set before the first engine initializes libfabric
    setenv("FI_PROVIDER", "tcp", 0);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
