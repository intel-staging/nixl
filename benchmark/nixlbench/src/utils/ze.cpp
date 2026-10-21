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

#include "ze.h"

#include <level_zero/ze_api.h>

#include <cstdint>
#include <mutex>
#include <vector>

namespace {

struct zeState {
    ze_result_t status = ZE_RESULT_ERROR_UNINITIALIZED;
    ze_context_handle_t context = nullptr;
    std::vector<ze_device_handle_t> devices;
};

/* Select the driver the way UCX's ZE transport does (GPU only, first driver
   that exposes devices), so UCX queries the memory through the same driver.
   Allocation lookup is driver-wide, so a separate context is fine.
   zeInit and zeDriverGet are deprecated since Level Zero 1.10, but UCX calls
   zeDriverGet in this process, and the spec makes mixing it with
   zeInitDrivers undefined behavior. */
ze_result_t
initState(zeState &state) {
    ze_result_t rc = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    if (rc != ZE_RESULT_SUCCESS) {
        return rc;
    }

    uint32_t driver_count = 0;
    rc = zeDriverGet(&driver_count, nullptr);
    if (rc != ZE_RESULT_SUCCESS) {
        return rc;
    }

    std::vector<ze_driver_handle_t> drivers(driver_count);
    rc = zeDriverGet(&driver_count, drivers.data());
    if (rc != ZE_RESULT_SUCCESS) {
        return rc;
    }

    for (auto driver : drivers) {
        uint32_t dev_count = 0;
        if (zeDeviceGet(driver, &dev_count, nullptr) != ZE_RESULT_SUCCESS || dev_count == 0) {
            continue;
        }

        state.devices.resize(dev_count);
        rc = zeDeviceGet(driver, &dev_count, state.devices.data());
        if (rc != ZE_RESULT_SUCCESS) {
            return rc;
        }

        ze_context_desc_t ctx_desc = {ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
        return zeContextCreate(driver, &ctx_desc, &state.context);
    }

    return ZE_RESULT_ERROR_UNINITIALIZED;
}

zeState &
getState() {
    static zeState state;
    static std::once_flag once;
    std::call_once(once, [] { state.status = initState(state); });
    return state;
}

ze_result_t
getDevice(const void *addr, ze_device_handle_t *device) {
    ze_memory_allocation_properties_t props = {};
    props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;

    ze_result_t rc = zeMemGetAllocProperties(getState().context, addr, &props, device);
    if (rc != ZE_RESULT_SUCCESS) {
        return rc;
    }
    if (props.type != ZE_MEMORY_TYPE_DEVICE || *device == nullptr) {
        return ZE_RESULT_ERROR_INVALID_ARGUMENT;
    }
    return ZE_RESULT_SUCCESS;
}

/* Ordinal of the first command queue group with compute engines. Such a
   group also accepts fills and copies; ordinal 0 is not guaranteed to be it. */
ze_result_t
getComputeOrdinal(ze_device_handle_t device, uint32_t *ordinal) {
    uint32_t group_count = 0;
    ze_result_t rc = zeDeviceGetCommandQueueGroupProperties(device, &group_count, nullptr);
    if (rc != ZE_RESULT_SUCCESS) {
        return rc;
    }

    std::vector<ze_command_queue_group_properties_t> groups(group_count);
    for (auto &group : groups) {
        group.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    }
    rc = zeDeviceGetCommandQueueGroupProperties(device, &group_count, groups.data());
    if (rc != ZE_RESULT_SUCCESS) {
        return rc;
    }

    for (uint32_t i = 0; i < group_count; ++i) {
        if ((groups[i].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) &&
            groups[i].numQueues > 0) {
            *ordinal = i;
            return ZE_RESULT_SUCCESS;
        }
    }
    return ZE_RESULT_ERROR_UNSUPPORTED_FEATURE;
}

/* Setup-time fills and copies only, never on the timed path: one synchronous
   immediate command list per call. */
template<typename appendFn>
ze_result_t
runOnDevice(ze_device_handle_t device, appendFn &&append) {
    ze_command_queue_desc_t queue_desc = {};
    queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    queue_desc.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
    ze_result_t rc = getComputeOrdinal(device, &queue_desc.ordinal);
    if (rc != ZE_RESULT_SUCCESS) {
        return rc;
    }

    ze_command_list_handle_t cmd_list = nullptr;
    rc = zeCommandListCreateImmediate(getState().context, device, &queue_desc, &cmd_list);
    if (rc != ZE_RESULT_SUCCESS) {
        return rc;
    }

    rc = append(cmd_list);
    if (rc == ZE_RESULT_SUCCESS) {
        rc = zeCommandListHostSynchronize(cmd_list, UINT64_MAX);
    }
    zeCommandListDestroy(cmd_list);
    return rc;
}

} // namespace

int
zeMalloc(void **addr, size_t buffer_size, int devid) {
    zeState &state = getState();
    if (state.status != ZE_RESULT_SUCCESS) {
        return state.status;
    }
    if (devid < 0 || (size_t)devid >= state.devices.size()) {
        std::cerr << "ZE: device " << devid << " out of range (" << state.devices.size()
                  << " visible)" << std::endl;
        return ZE_RESULT_ERROR_INVALID_ARGUMENT;
    }

    /* Make the allocation DMA-BUF exportable. UCX's ze_copy MD registers VRAM
       with the NIC through the fd from zeMemGetAllocProperties. Without this
       descriptor, when only one device is visible, the driver can carve small
       buffers out of a shared device memory pool, so the fd does not map this
       buffer and RDMA misses it while still reporting success. Exportable
       memory also bypasses compression, which the KMD does not support for
       shared allocations. */
    ze_external_memory_export_desc_t export_desc = {};
    export_desc.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC;
    export_desc.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;

    ze_device_mem_alloc_desc_t alloc_desc = {};
    alloc_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    alloc_desc.pNext = &export_desc;
    return zeMemAllocDevice(state.context, &alloc_desc, buffer_size, 0, state.devices[devid], addr);
}

int
zeFree(void *addr) {
    return zeMemFree(getState().context, addr);
}

int
zeMemset(void *addr, int val, size_t count) {
    ze_device_handle_t device = nullptr;
    ze_result_t rc = getDevice(addr, &device);
    if (rc != ZE_RESULT_SUCCESS) {
        return rc;
    }

    const uint8_t pattern = (uint8_t)val;
    return runOnDevice(device, [&](ze_command_list_handle_t cmd_list) {
        return zeCommandListAppendMemoryFill(
            cmd_list, addr, &pattern, sizeof(pattern), count, nullptr, 0, nullptr);
    });
}

int
zeMemcpyDeviceToHost(void *dest, const void *src, size_t count) {
    ze_device_handle_t device = nullptr;
    ze_result_t rc = getDevice(src, &device);
    if (rc != ZE_RESULT_SUCCESS) {
        return rc;
    }

    return runOnDevice(device, [&](ze_command_list_handle_t cmd_list) {
        return zeCommandListAppendMemoryCopy(cmd_list, dest, src, count, nullptr, 0, nullptr);
    });
}
