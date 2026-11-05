/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 * Mocked tests for the verbs provider's connection-setup behaviour, needing no verbs hardware:
 * - a rail opened with auto_progress requests FI_PROGRESS_AUTO and FI_THREAD_SAFE, keeps them
 *   on the retry without FI_HMEM, and fails with a clear error when the provider refuses them;
 * - progressActiveRails() polls every rail for verbs, and only rail 0 plus the active rails
 *   otherwise;
 * - canCarryDeviceMemory() requires FI_HMEM, and a provider that carries it, for Level Zero.
 */

#include "libfabric/libfabric_rail_manager.h"
#include "libfabric/libfabric_common.h"
#include "common/nixl_log.h"
#include "libfabric_mock_stubs.h"

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t k_num_rails = 4;

int failures = 0;

void
check(bool cond, const std::string &what) {
    if (cond) {
        NIXL_INFO << "  PASS: " << what;
    } else {
        NIXL_ERROR << "  FAIL: " << what;
        ++failures;
    }
}

// --- fi_getinfo mock ---

// How the mock provider answers hints that request FI_PROGRESS_AUTO.
enum class AutoMode {
    Grant, // returns entries with FI_PROGRESS_AUTO
    Reject, // returns no entries (FI_ENODATA), as libfabric filters
    Manual, // returns entries with FI_PROGRESS_MANUAL regardless
};

std::string mock_prov_name = "efa";
std::string mock_domain_prefix = "efa_";
AutoMode mock_auto_mode = AutoMode::Grant;
bool mock_reject_fi_hmem = false;

struct RecordedHints {
    uint64_t caps;
    fi_threading threading;
    fi_progress control_progress;
    fi_progress data_progress;
    std::string prov_name;
};

std::vector<RecordedHints> recorded_hints;

bool
requestsAuto(const RecordedHints &h) {
    return h.threading == FI_THREAD_SAFE && h.control_progress == FI_PROGRESS_AUTO &&
        h.data_progress == FI_PROGRESS_AUTO;
}

// --- CQ polling counters, one per CQ in creation order (one CQ per rail) ---

std::vector<fid_cq *> cq_order;
std::map<fid_cq *, size_t> cq_reads;

int
counting_cq_open(fid_domain *domain, fi_cq_attr *attr, fid_cq **cq, void *context) {
    const int ret = fi_cq_open_stub(domain, attr, cq, context);
    if (ret == 0) {
        cq_order.push_back(*cq);
    }
    return ret;
}

ssize_t
counting_cq_readfrom(fid_cq *cq, void * /*buf*/, size_t /*count*/, fi_addr_t * /*src_addr*/) {
    ++cq_reads[cq];
    return -FI_EAGAIN;
}

ssize_t
counting_cq_readerr(fid_cq * /*cq*/, fi_cq_err_entry * /*buf*/, uint64_t /*flags*/) {
    return -FI_EAGAIN;
}

size_t
readsOfRail(size_t rail) {
    if (rail >= cq_order.size()) {
        return 0;
    }
    const auto it = cq_reads.find(cq_order[rail]);
    return (it != cq_reads.end()) ? it->second : 0;
}

void
useProvider(const std::string &prov_name, const std::string &domain_prefix) {
    mock_prov_name = prov_name;
    mock_domain_prefix = domain_prefix;
    mock_auto_mode = AutoMode::Grant;
    mock_reject_fi_hmem = false;
    recorded_hints.clear();
    cq_order.clear();
    cq_reads.clear();
}

} // namespace

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
                  const struct fi_info *hints,
                  struct fi_info **info) {
    *info = nullptr;
    bool wants_auto = false;
    if (hints != nullptr && hints->domain_attr != nullptr) {
        RecordedHints rec{hints->caps,
                          hints->domain_attr->threading,
                          hints->domain_attr->control_progress,
                          hints->domain_attr->data_progress,
                          (hints->fabric_attr && hints->fabric_attr->prov_name) ?
                              hints->fabric_attr->prov_name :
                              ""};
        wants_auto = requestsAuto(rec);
        recorded_hints.push_back(rec);
    }
    if (hints != nullptr && mock_reject_fi_hmem && (hints->caps & FI_HMEM)) {
        return -FI_ENODATA;
    }
    if (wants_auto && mock_auto_mode == AutoMode::Reject) {
        return -FI_ENODATA;
    }

    fi_info *head = mock_fi_info_chain(k_num_rails, 400ull * NIXL_LIBFABRIC_GIGA);
    size_t i = 0;
    for (fi_info *fi = head; fi != nullptr; fi = fi->next, ++i) {
        free(fi->fabric_attr->prov_name);
        fi->fabric_attr->prov_name = strdup(mock_prov_name.c_str());
        free(fi->domain_attr->name);
        fi->domain_attr->name = strdup((mock_domain_prefix + std::to_string(i)).c_str());
        const bool granted = wants_auto && mock_auto_mode == AutoMode::Grant;
        fi->domain_attr->threading = granted ? FI_THREAD_SAFE : FI_THREAD_COMPLETION;
        fi->domain_attr->control_progress = granted ? FI_PROGRESS_AUTO : FI_PROGRESS_MANUAL;
        fi->domain_attr->data_progress = granted ? FI_PROGRESS_AUTO : FI_PROGRESS_MANUAL;
    }
    *info = head;
    return 0;
}

extern "C" int
__wrap_fi_fabric(struct fi_fabric_attr * /*attr*/, struct fid_fabric **fabric, void * /*context*/) {
    *fabric = mock_fabric_create();
    return 0;
}

namespace {

/****************************************
 * 1. Rail auto-progress negotiation
 *****************************************/

bool
railThrowsAutoProgressError() {
    try {
        nixlLibfabricRail rail("mlx5_0", "verbs", 0, FI_HMEM_SYSTEM, false, true);
    }
    catch (const std::runtime_error &e) {
        return std::string(e.what()).find("FI_PROGRESS_AUTO") != std::string::npos;
    }
    return false;
}

void
testRailAutoProgress() {
    NIXL_INFO << "1. Testing rail auto-progress negotiation";

    useProvider("verbs;ofi_rxm", "mlx5_");
    bool constructed = false;
    try {
        nixlLibfabricRail rail("mlx5_0", "verbs", 0, FI_HMEM_SYSTEM, false, true);
        constructed = true;
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "   " << e.what();
    }
    check(constructed, "a rail is created when the provider grants FI_PROGRESS_AUTO");
    check(!recorded_hints.empty() && requestsAuto(recorded_hints.front()),
          "the rail requests FI_PROGRESS_AUTO with FI_THREAD_SAFE");
    check(!recorded_hints.empty() && recorded_hints.front().prov_name == "verbs",
          "the rail names verbs in its hints");

    useProvider("verbs;ofi_rxm", "mlx5_");
    mock_reject_fi_hmem = true;
    try {
        nixlLibfabricRail rail("mlx5_0", "verbs", 0, FI_HMEM_SYSTEM, false, true);
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "   " << e.what();
    }
    check(recorded_hints.size() >= 2 && requestsAuto(recorded_hints[0]) &&
              requestsAuto(recorded_hints[1]) && (recorded_hints[1].caps & FI_HMEM) == 0,
          "FI_PROGRESS_AUTO and FI_THREAD_SAFE survive the retry without FI_HMEM");

    useProvider("verbs;ofi_rxm", "mlx5_");
    mock_auto_mode = AutoMode::Reject;
    check(railThrowsAutoProgressError(),
          "a provider that filters out FI_PROGRESS_AUTO fails rail creation with that diagnostic");

    useProvider("verbs;ofi_rxm", "mlx5_");
    mock_auto_mode = AutoMode::Manual;
    check(railThrowsAutoProgressError(),
          "a provider that answers with manual progress fails rail creation with that diagnostic");

    useProvider("verbs;ofi_rxm", "mlx5_");
    { nixlLibfabricRail rail("mlx5_0", "verbs", 0, FI_HMEM_SYSTEM, false, false); }
    check(!recorded_hints.empty() && !requestsAuto(recorded_hints.front()) &&
              recorded_hints.front().threading == FI_THREAD_COMPLETION,
          "without auto_progress the rail keeps FI_THREAD_COMPLETION and default progress");
}

/****************************************
 * 2. Which rails progressActiveRails() polls
 *****************************************/

void
testProgressAllRails() {
    NIXL_INFO << "2. Testing progressActiveRails() rail coverage";

    // verbs, no progress thread: rails open with FI_PROGRESS_AUTO and every rail is polled.
    useProvider("verbs;ofi_rxm", "mlx5_");
    {
        nixlLibfabricRailManager mgr(0, /*progress_thread_enabled=*/false);
        check(mgr.getNumRails() == k_num_rails && cq_order.size() == k_num_rails,
              "verbs: one rail and one CQ per mock NIC");
        check(mgr.needsAllRailProgress(), "verbs: the manager progresses every rail");
        bool all_auto = true;
        for (const auto &h : recorded_hints) {
            if (h.prov_name == "verbs") {
                all_auto = all_auto && requestsAuto(h);
            }
        }
        check(all_auto, "verbs without a progress thread: every rail requests FI_PROGRESS_AUTO");

        mgr.clearActiveRails();
        cq_reads.clear();
        mgr.progressActiveRails();
        bool all_polled = true;
        for (size_t rail = 0; rail < k_num_rails; ++rail) {
            all_polled = all_polled && readsOfRail(rail) == 1;
        }
        check(all_polled,
              "verbs: with no active rails, every rail (including rails 1-3) is polled");
    }

    // efa: only rail 0 and the active rails are polled.
    useProvider("efa", "efa_");
    {
        nixlLibfabricRailManager mgr(0, /*progress_thread_enabled=*/false);
        check(mgr.getNumRails() == k_num_rails && cq_order.size() == k_num_rails,
              "efa: one rail and one CQ per mock NIC");
        check(!mgr.needsAllRailProgress(), "efa: the manager progresses active rails only");
        bool any_auto = false;
        for (const auto &h : recorded_hints) {
            any_auto = any_auto || requestsAuto(h);
        }
        check(!any_auto, "efa: no rail requests FI_PROGRESS_AUTO");

        mgr.clearActiveRails();
        cq_reads.clear();
        mgr.progressActiveRails();
        check(readsOfRail(0) == 1 && readsOfRail(1) == 0 && readsOfRail(2) == 0 &&
                  readsOfRail(3) == 0,
              "efa: with no active rails only rail 0 is polled");

        mgr.incRailActive(2);
        cq_reads.clear();
        mgr.progressActiveRails();
        check(readsOfRail(0) == 1 && readsOfRail(1) == 0 && readsOfRail(2) == 1 &&
                  readsOfRail(3) == 0,
              "efa: rail 0 and the active rail 2 are polled");
        mgr.clearActiveRails();
    }
}

/****************************************
 * 3. Which rails can carry device memory
 *****************************************/

bool
carries(const std::string &prov_name, enum fi_hmem_iface runtime, bool with_hmem) {
    useProvider(prov_name, prov_name + "_");
    mock_reject_fi_hmem = !with_hmem;
    nixlLibfabricRail rail(prov_name + "_0", prov_name, 0, runtime, true, false);
    return rail.canCarryDeviceMemory();
}

void
testCanCarryDeviceMemory() {
    NIXL_INFO << "3. Testing canCarryDeviceMemory()";

    check(carries("verbs", FI_HMEM_ZE, true), "verbs with FI_HMEM carries Level Zero memory");
    check(!carries("verbs", FI_HMEM_ZE, false),
          "verbs without FI_HMEM does not carry Level Zero memory (no Intel peer-memory module)");
    check(!carries("efa", FI_HMEM_ZE, true),
          "efa does not carry Level Zero memory, even with FI_HMEM");
    check(carries("efa", FI_HMEM_CUDA, true), "efa with FI_HMEM carries CUDA memory");
    check(carries("verbs", FI_HMEM_CUDA, false),
          "verbs without FI_HMEM still carries CUDA memory through NIC peer memory");
    check(!carries("tcp", FI_HMEM_CUDA, false), "tcp without FI_HMEM does not carry device memory");
    check(carries("tcp", FI_HMEM_CUDA, true), "tcp with FI_HMEM carries device memory");
    check(carries("tcp", FI_HMEM_ZE, true), "tcp with FI_HMEM carries Level Zero memory by copy");
    check(!carries("tcp", FI_HMEM_ZE, false),
          "tcp without FI_HMEM does not carry Level Zero memory");
}

} // namespace

int
main() {
    NIXL_INFO << "=== Testing verbs connection setup with mocked libfabric ===";

    domain_ops_stub.cq_open = counting_cq_open;
    cq_ops_stub.size = sizeof(fi_ops_cq);
    cq_ops_stub.readfrom = counting_cq_readfrom;
    cq_ops_stub.readerr = counting_cq_readerr;

    try {
        testRailAutoProgress();
        testProgressAllRails();
        testCanCarryDeviceMemory();
    }
    catch (const std::exception &e) {
        NIXL_ERROR << "Unexpected exception: " << e.what();
        ++failures;
    }

    if (failures != 0) {
        NIXL_ERROR << "=== " << failures << " check(s) FAILED ===";
        return 1;
    }
    NIXL_INFO << "=== Test completed successfully! ===";
    return 0;
}
