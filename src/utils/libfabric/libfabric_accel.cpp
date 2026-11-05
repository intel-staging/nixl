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

constexpr uint16_t k_intel_pci_vendor = 0x8086;
// PCI base classes an Intel GPU reports: display controller, processing accelerator.
constexpr unsigned k_pci_class_display = 0x03;
constexpr unsigned k_pci_class_processing_accel = 0x12;

// Discrete-GPU addresses from the Level Zero driver, in hwloc's format. Cached: the driver's
// device list does not change, and the caller queries it for every PCI device in the topology.
// Never freed, so a late caller during exit() does not read a destroyed set.
const std::set<std::string> &
zeAccelBdfs() {
    static const std::set<std::string> *bdfs = []() {
        auto *out_ptr = new std::set<std::string>();
        std::set<std::string> &out = *out_ptr;

        const size_t count = nfi_hmem_device_bdfs(FI_HMEM_ZE, nullptr, 0);
        if (count == 0) {
            return out_ptr;
        }

        std::vector<struct nfi_hmem_bus_id> ids(count);
        const size_t written = nfi_hmem_device_bdfs(FI_HMEM_ZE, ids.data(), ids.size());
        for (size_t i = 0; i < std::min(written, ids.size()); ++i) {
            out.insert(accelBusIdString(ids[i], NFI_HMEM_BUS_ACCELERATOR));
        }
        return out_ptr;
    }();
    return *bdfs;
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

    // An address match alone is not enough: a topology loaded from XML (HWLOC_XMLFILE) can put
    // another vendor's device at a local GPU's address. Also keeps Level Zero uninitialised on
    // hosts with no Intel display or processing-accelerator device.
    const unsigned base_class = obj->attr->pcidev.class_id >> 8;
    if (obj->attr->pcidev.vendor_id != k_intel_pci_vendor ||
        (base_class != k_pci_class_display && base_class != k_pci_class_processing_accel)) {
        return false;
    }

    const std::set<std::string> &enumerated = zeAccelBdfs();
    if (enumerated.empty()) {
        return false;
    }

    struct nfi_hmem_bus_id bus;
    bus.domain = static_cast<uint32_t>(obj->attr->pcidev.domain);
    bus.bus = static_cast<uint8_t>(obj->attr->pcidev.bus);
    bus.slot = static_cast<uint8_t>(obj->attr->pcidev.dev);
    bus.func = static_cast<uint8_t>(obj->attr->pcidev.func);

    return enumerated.count(accelBusIdString(bus, NFI_HMEM_BUS_ACCELERATOR)) > 0;
}
