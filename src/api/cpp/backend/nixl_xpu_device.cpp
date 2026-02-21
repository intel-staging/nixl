/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 Intel Corporation. All rights reserved.
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

#ifdef HAVE_ZE

#include "nixl_xpu_device.h"
#include "common/nixl_log.h"

#include <cstdio>

std::vector<XpuDevice>
XpuDevice::detectDevices() {
    std::vector<XpuDevice> result;

    ze_result_t rc = zeInit(0);
    if (rc != ZE_RESULT_SUCCESS) {
        NIXL_WARN << "zeInit failed (rc=" << rc << "), Intel XPU support unavailable";
        return result;
    }

    uint32_t driver_count = 0;
    rc = zeDriverGet(&driver_count, nullptr);
    if (rc != ZE_RESULT_SUCCESS || driver_count == 0) {
        NIXL_INFO << "No Level Zero drivers found";
        return result;
    }

    std::vector<ze_driver_handle_t> drivers(driver_count);
    rc = zeDriverGet(&driver_count, drivers.data());
    if (rc != ZE_RESULT_SUCCESS) {
        NIXL_ERROR << "zeDriverGet (enumerate) failed (rc=" << rc << ")";
        return result;
    }

    for (uint32_t dri = 0; dri < driver_count; ++dri) {
        ze_driver_handle_t drv = drivers[dri];

        uint32_t device_count = 0;
        rc = zeDeviceGet(drv, &device_count, nullptr);
        if (rc != ZE_RESULT_SUCCESS || device_count == 0)
            continue;

        std::vector<ze_device_handle_t> devices(device_count);
        rc = zeDeviceGet(drv, &device_count, devices.data());
        if (rc != ZE_RESULT_SUCCESS)
            continue;

        // Create one shared context per driver covering all its devices.
        ze_context_desc_t ctx_desc = { ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0 };
        ze_context_handle_t ctx = nullptr;
        rc = zeContextCreate(drv, &ctx_desc, &ctx);
        if (rc != ZE_RESULT_SUCCESS) {
            NIXL_WARN << "zeContextCreate failed for driver " << dri << " (rc=" << rc << ")";
            continue;
        }

        for (uint32_t i = 0; i < device_count; ++i) {
            XpuDevice xpu;
            xpu.device        = devices[i];
            xpu.context       = ctx;
            xpu.driver_index  = dri;
            xpu.device_index  = i;

            // Query PCI BDF via the ZE extension property.
            ze_pci_ext_properties_t pci_props = {};
            pci_props.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
            ze_result_t pci_rc = zeDevicePciGetPropertiesExt(devices[i], &pci_props);
            if (pci_rc == ZE_RESULT_SUCCESS) {
                char bdf[32];
                std::snprintf(bdf, sizeof(bdf), "%x:%02x:%02x.%x",
                              pci_props.address.domain,
                              pci_props.address.bus,
                              pci_props.address.device,
                              pci_props.address.function);
                xpu.pci_bdf = bdf;
            } else {
                NIXL_WARN << "zeDevicePciGetPropertiesExt failed for device " << i
                          << " (rc=" << pci_rc << ")";
            }

            NIXL_INFO << "Detected Intel XPU device[" << i << "] driver=" << dri
                      << " pci_bdf=" << xpu.pci_bdf;
            result.push_back(xpu);
        }
    }

    return result;
}

ze_device_handle_t
XpuDevice::getDeviceForPtr(ze_context_handle_t context,
                           const std::vector<XpuDevice> &devices,
                           const void *ptr) {
    if (!context || !ptr)
        return nullptr;

    ze_memory_allocation_properties_t alloc_props = {};
    alloc_props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
    ze_device_handle_t alloc_device = nullptr;

    ze_result_t rc = zeMemGetAllocProperties(context, ptr, &alloc_props, &alloc_device);
    if (rc != ZE_RESULT_SUCCESS) {
        NIXL_DEBUG << "zeMemGetAllocProperties failed (rc=" << rc << ")";
        return nullptr;
    }

    // Verify the returned handle belongs to our tracked device list.
    for (const auto &xpu : devices) {
        if (xpu.device == alloc_device)
            return alloc_device;
    }

    return nullptr;
}

#endif // HAVE_ZE
