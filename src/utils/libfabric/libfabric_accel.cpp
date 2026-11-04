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

#include "libfabric_accel.h"
#include "common/nixl_log.h"

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <set>
#include <vector>

namespace {

/** @brief Whether @p obj is a PCI device of @p vendor_id in the display controller class. */
bool
isDisplayClassPciDev(hwloc_obj_t obj, uint16_t vendor_id) {
    if (!obj || obj->type != HWLOC_OBJ_PCI_DEVICE || obj->attr->pcidev.vendor_id != vendor_id) {
        return false;
    }
    // 0x300-0x3ff are display controllers; 0x302 is a 3D controller (compute GPU)
    const uint16_t class_id = obj->attr->pcidev.class_id;
    return (class_id >= 0x300 && class_id < 0x400);
}

bool
isNvidiaAccel(hwloc_obj_t obj) {
    return isDisplayClassPciDev(obj, 0x10de);
}

bool
isAmdAccel(hwloc_obj_t obj) {
    if (!obj || obj->type != HWLOC_OBJ_PCI_DEVICE || obj->attr->pcidev.vendor_id != 0x1002) {
        return false;
    }
    // MI300X uses class 0x1200 (processing accelerator), consumer GPUs use 0x300-0x3ff
    const uint16_t class_id = obj->attr->pcidev.class_id;
    return (class_id >= 0x300 && class_id < 0x400) || (class_id >= 0x1200 && class_id < 0x1300);
}

bool
isNeuronAccel(hwloc_obj_t obj) {
    // Amazon vendor ID is 0x1d0f
    if (!obj || obj->type != HWLOC_OBJ_PCI_DEVICE || obj->attr->pcidev.vendor_id != 0x1d0f) {
        return false;
    }
    static const uint16_t NEURON_DEVICE_IDS[] = {
        0x7264, // INF2
        0x7164, // TRN1
        0x7364, // TRN2
        0x7564, // TRN3_DEVICE_0
        0x7565, // TRN3_DEVICE_1
    };
    return std::find(std::begin(NEURON_DEVICE_IDS),
                     std::end(NEURON_DEVICE_IDS),
                     obj->attr->pcidev.device_id) != std::end(NEURON_DEVICE_IDS);
}

// Discrete-GPU addresses from the Level Zero driver, in hwloc's format. Cached: the driver's
// device list does not change, and the caller queries it for every PCI device in the topology.
const std::set<std::string> &
zeAccelBdfs() {
    static const std::set<std::string> bdfs = []() {
        std::set<std::string> out;

        const size_t count = nfi_hmem_device_bdfs(FI_HMEM_ZE, nullptr, 0);
        if (count == 0) {
            return out;
        }

        std::vector<struct nfi_hmem_bus_id> ids(count);
        const size_t written = nfi_hmem_device_bdfs(FI_HMEM_ZE, ids.data(), ids.size());
        for (size_t i = 0; i < std::min(written, ids.size()); ++i) {
            out.insert(accelBusIdString(ids[i], NFI_HMEM_BUS_ACCELERATOR));
        }
        return out;
    }();
    return bdfs;
}

void
logBridge(enum nfi_hmem_log_level level, const char *msg) {
    switch (level) {
    case NFI_HMEM_LOG_ERROR:
        NIXL_ERROR << "hmem: " << msg;
        break;
    case NFI_HMEM_LOG_WARN:
        NIXL_WARN << "hmem: " << msg;
        break;
    case NFI_HMEM_LOG_INFO:
        NIXL_INFO << "hmem: " << msg;
        break;
    case NFI_HMEM_LOG_DEBUG:
    default:
        NIXL_DEBUG << "hmem: " << msg;
        break;
    }
}

} // namespace

void
nixlAccelInstallLogBridge() {
    static std::once_flag once;
    std::call_once(once, []() { nfi_hmem_set_log_fn(logBridge); });
}

std::string
accelBusIdString(const struct nfi_hmem_bus_id &bus, enum nfi_hmem_bus_kind kind) {
    if (kind == NFI_HMEM_BUS_NONE) {
        return "";
    }

    char buf[32];
    snprintf(buf,
             sizeof(buf),
             "%x:%02x:%02x.%x",
             static_cast<unsigned>(bus.domain),
             static_cast<unsigned>(bus.bus),
             static_cast<unsigned>(bus.slot),
             static_cast<unsigned>(bus.func));
    return std::string(buf);
}

std::string
accelBusIdString(const struct nfi_hmem_info &info) {
    return accelBusIdString(info.bus, info.bus_kind);
}

bool
accelIsLevelZeroDevice(hwloc_obj_t obj) {
    if (!obj || obj->type != HWLOC_OBJ_PCI_DEVICE) {
        return false;
    }

    const std::set<std::string> &enumerated = zeAccelBdfs();
    if (enumerated.empty()) {
        return false;
    }

    struct nfi_hmem_bus_id bus;
    bus.domain = static_cast<uint16_t>(obj->attr->pcidev.domain);
    bus.bus = static_cast<uint8_t>(obj->attr->pcidev.bus);
    bus.slot = static_cast<uint8_t>(obj->attr->pcidev.dev);
    bus.func = static_cast<uint8_t>(obj->attr->pcidev.func);

    return enumerated.count(accelBusIdString(bus, NFI_HMEM_BUS_ACCELERATOR)) > 0;
}

const std::vector<enum fi_hmem_iface> &
accelHwlocIfaces() {
    static const std::vector<enum fi_hmem_iface> ifaces = {
        FI_HMEM_CUDA, FI_HMEM_ROCR, FI_HMEM_ZE, FI_HMEM_NEURON};

    return ifaces;
}

enum fi_hmem_iface
accelIfaceOfHwlocObj(hwloc_obj_t obj) {
    if (isNvidiaAccel(obj)) {
        return FI_HMEM_CUDA;
    }
    if (isAmdAccel(obj)) {
        return FI_HMEM_ROCR;
    }
    if (isNeuronAccel(obj)) {
        return FI_HMEM_NEURON;
    }
    if (accelIsLevelZeroDevice(obj)) {
        return FI_HMEM_ZE;
    }
    return FI_HMEM_SYSTEM;
}

bool
accelIfaceIsGroupable(enum fi_hmem_iface iface) {
    return iface == FI_HMEM_CUDA || iface == FI_HMEM_ROCR || iface == FI_HMEM_ZE;
}
