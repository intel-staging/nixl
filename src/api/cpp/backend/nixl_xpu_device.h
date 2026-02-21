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
#ifndef NIXL_SRC_API_CPP_BACKEND_NIXL_XPU_DEVICE_H
#define NIXL_SRC_API_CPP_BACKEND_NIXL_XPU_DEVICE_H

#ifdef HAVE_ZE
#include <level_zero/ze_api.h>
#include <vector>
#include <string>
#include <cstdint>

/**
 * @brief Wraps a Level Zero device + context for Intel XPU support.
 *
 * Mirrors the role of CudaDevice in the CUDA path: holds a device handle,
 * a context, the device ordinal, and the PCIe BDF for topology matching.
 */
class XpuDevice {
public:
    ze_device_handle_t  device;
    ze_context_handle_t context;
    uint32_t            driver_index; // index of the Level Zero driver
    uint32_t            device_index; // ordinal within its driver
    // PCI BDF normalised as "domain:bus:dev.func" (e.g. "0:3a:00.0")
    std::string         pci_bdf;

    XpuDevice() : device(nullptr), context(nullptr), driver_index(0), device_index(0) {}

    /**
     * @brief Discover all Level Zero devices and return one XpuDevice per device.
     *
     * Calls zeInit / zeDriverGet / zeDeviceGet, then
     * zeDevicePciGetPropertiesExt to populate pci_bdf.
     * Returns an empty vector if Level Zero is unavailable.
     */
    static std::vector<XpuDevice> detectDevices();

    /**
     * @brief Given a device-allocated pointer, return the ze_device_handle_t
     * that owns it using zeMemGetAllocProperties.
     *
     * @param context  A valid Level Zero context.
     * @param devices  The list produced by detectDevices().
     * @param ptr      The device pointer to query.
     * @return the matching handle, or nullptr if not found / error.
     */
    static ze_device_handle_t getDeviceForPtr(ze_context_handle_t context,
                                              const std::vector<XpuDevice> &devices,
                                              const void *ptr);
};

#endif // HAVE_ZE
#endif // NIXL_SRC_API_CPP_BACKEND_NIXL_XPU_DEVICE_H
