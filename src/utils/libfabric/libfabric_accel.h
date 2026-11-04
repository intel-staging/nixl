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
#ifndef NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_ACCEL_H
#define NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_ACCEL_H

#include "nfi_hmem.h"

#include <hwloc.h>
#include <string>
#include <vector>

/*
 * nixl-side helpers around nfi_hmem.h: matching accelerator devices against the hwloc topology and
 * routing the shim's diagnostics into NIXL_LOG.
 */

/** @brief Routes nfi_hmem diagnostics into NIXL_LOG. Idempotent. */
void
nixlAccelInstallLogBridge();

/**
 * @brief Formats a PCI address as hwloc does ("%x:%02x:%02x.%x").
 * @return Empty string when @p kind is NFI_HMEM_BUS_NONE.
 */
std::string
accelBusIdString(const struct nfi_hmem_bus_id &bus, enum nfi_hmem_bus_kind kind);

/** @brief @ref accelBusIdString for the address inside an info block. */
std::string
accelBusIdString(const struct nfi_hmem_info &info);

/**
 * @brief Whether @p obj is a discrete GPU enumerated by the Level Zero driver.
 *
 * Matched against the driver's device list because PCI attributes cannot tell a discrete Intel GPU
 * from integrated graphics (same vendor and class), and compute-only parts have no display class.
 * False when Level Zero is unavailable or @p obj is not a PCI device.
 */
bool
accelIsLevelZeroDevice(hwloc_obj_t obj);

/** @brief The interfaces @ref accelIfaceOfHwlocObj can attribute a device to. */
const std::vector<enum fi_hmem_iface> &
accelHwlocIfaces();

/**
 * @brief The interface of the accelerator vendor that owns @p obj, or FI_HMEM_SYSTEM if none.
 *
 * NVIDIA and AMD GPUs are matched by PCI vendor and class, Neuron devices by vendor and device ID,
 * and Intel GPUs by @ref accelIsLevelZeroDevice.
 */
enum fi_hmem_iface
accelIfaceOfHwlocObj(hwloc_obj_t obj);

/**
 * @brief Whether @p iface's devices take part in PCIe-proximity NIC grouping.
 *
 * False for Neuron: a Trainium card carries its own EFA NIC, so Neuron memory is mapped to NICs
 * through the NIC's address instead.
 */
bool
accelIfaceIsGroupable(enum fi_hmem_iface iface);

#endif // NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_ACCEL_H
