/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <iostream>
#include <mutex>
#include "synapse_utils.h"

static bool device_initialized = false;
static std::mutex mtx;
static synDeviceId deviceHandle;
static synStreamHandle stream;

namespace Synapseaiutils {
static void
check(int ret, const char *msg) {
    if (ret) {
        fprintf(stderr, "%s: %s(%d)\n", msg, "failed", -ret);
        exit(1);
    }
}

int
init_synapse_device() {
    std::lock_guard<std::mutex> lock(mtx);
    auto env = std::getenv("HLS_MODULE_ID");
    int module_id = 0;
    if (env != nullptr) {
        module_id = std::stoi(env);
    }
    if (device_initialized) return 0;
    check(synInitialize(), "synInitialize");
    check(synDeviceAcquireByModuleId(&deviceHandle, module_id), "synDeviceAcquire");
    device_initialized = true;
    check(synStreamCreateGeneric(&stream, deviceHandle, 0), "synStreamCreateGeneric");
    return 0;
}

synDeviceId
get_device_handle() {
    return deviceHandle;
}

uint64_t
allocate_synapse_memory(size_t len, void *host_buffer) {
    uint64_t device_buffer;
    std::lock_guard<std::mutex> lock(mtx);
    if (!device_initialized) {
        fprintf(stderr, "%s\n", "device nor initialized");
        exit(1);
    }

    check(synDeviceMalloc(deviceHandle, len, 0x0, 0, &device_buffer), "synDeviceMalloc");
    check(synHostMap(deviceHandle, len, host_buffer), "synHostMap");
    check(synMemCopyAsync(stream, (uint64_t)host_buffer, len, device_buffer, HOST_TO_DRAM),
          "synMemCopyAsync");
    check(synStreamSynchronize(stream), "synStreamSynchronize");
    std::cout << "allocate_synapse_memory" << "device buffer::" << device_buffer
              << " host buffer::" << host_buffer << " Len::" << len << std::endl;
    check(synHostUnmap(deviceHandle, host_buffer), "synHostUnmap");
    return device_buffer;
}

void
free_synapse_memory(uint64_t ptr) {
    std::lock_guard<std::mutex> lock(mtx);
    if (!device_initialized) fprintf(stderr, "%s\n", "device nor initialized");
    // cleanup Synapse resources
    check(synDeviceFree(deviceHandle, ptr, 0), "synDeviceFree");
}

void
deinit_synapse_device() {
    std::lock_guard<std::mutex> lock(mtx);
    if (!device_initialized) {
        fprintf(stderr, "%s\n", "device nor initialized");
        exit(1);
    }
    check(synStreamDestroy(stream), "synStreamDestroy");
    check(synDeviceRelease(deviceHandle), "synDeviceRelease");
    check(synDestroy(), "synDestroy");
    device_initialized = false;
}

void
copy_from_device_buffer(uint64_t device_buffer, void *host_buffer, size_t len) {
    std::lock_guard<std::mutex> lock(mtx);
    if (!device_initialized) {
        fprintf(stderr, "%s\n", "device nor initialized");
        exit(1);
    }
    check(synHostMap(deviceHandle, len, host_buffer), "synHostMap");
    check(synMemCopyAsync(stream, device_buffer, len, (uint64_t)host_buffer, DRAM_TO_HOST),
          "synMemCopyAsync");
    check(synStreamSynchronize(stream), "synStreamSynchronize");
    check(synHostUnmap(deviceHandle, host_buffer), "synHostUnmap");
}
} // namespace Synapseaiutils
