// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2026 Intel Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef NIXL_TEST_GTEST_GPU_UTILS_H
#define NIXL_TEST_GTEST_GPU_UTILS_H

#include <iostream>
#include <cstdlib>

#if defined(HAVE_CUDA)
#include <cuda_runtime.h>
#include <cuda.h>

inline void
checkCudaError(cudaError_t result, const char *message) {
    if (result != cudaSuccess) {
        std::cerr << message << " (Error code: " << result << " - " << cudaGetErrorString(result)
                  << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}
#endif

#if defined(HAVE_ROCM)
#include <hip/hip_runtime.h>

inline void
checkHipError(hipError_t result, const char *message) {
    if (result != hipSuccess) {
        std::cerr << message << " (Error code: " << result << " - " << hipGetErrorString(result)
                  << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}
#endif

// HAVE_ZE is defined project-wide, so it can coexist with HAVE_CUDA or HAVE_ROCM. Level
// Zero backs the helpers below only when neither of those is selected.
#if defined(HAVE_ZE) && !defined(HAVE_CUDA) && !defined(HAVE_ROCM)
#include <level_zero/ze_api.h>

#include <algorithm>
#include <cstdint>
#include <vector>

inline void
checkZeError(ze_result_t result, const char *message) {
    if (result != ZE_RESULT_SUCCESS) {
        std::cerr << message << " (Error code: 0x" << std::hex << result << std::dec << ")"
                  << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Level Zero has no implicit primary context, unlike CUDA, so the tests share one
// process-wide context on the first device of the first driver.
struct zeTestDevice {
    ze_context_handle_t context = nullptr;
    ze_device_handle_t device = nullptr;
    // Ordinal of the first command queue group that supports copies.
    uint32_t copyOrdinal = 0;
    // Why the device is unusable, or nullptr if it is usable.
    const char *error = nullptr;

    zeTestDevice() {
        if (zeInit(ZE_INIT_FLAG_GPU_ONLY) != ZE_RESULT_SUCCESS) {
            error = "zeInit failed";
            return;
        }

        uint32_t count = 1;
        ze_driver_handle_t driver = nullptr;
        if (zeDriverGet(&count, &driver) != ZE_RESULT_SUCCESS || count == 0) {
            error = "no Level Zero driver";
            return;
        }

        count = 1;
        if (zeDeviceGet(driver, &count, &device) != ZE_RESULT_SUCCESS || count == 0) {
            device = nullptr;
            error = "no Level Zero device";
            return;
        }

        ze_device_external_memory_properties_t props{};
        props.stype = ZE_STRUCTURE_TYPE_DEVICE_EXTERNAL_MEMORY_PROPERTIES;
        if (zeDeviceGetExternalMemoryProperties(device, &props) != ZE_RESULT_SUCCESS ||
            !(props.memoryAllocationExportTypes & ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF)) {
            error = "device cannot export allocations as DMA-BUF";
            return;
        }

        uint32_t group_count = 0;
        if (zeDeviceGetCommandQueueGroupProperties(device, &group_count, nullptr) !=
            ZE_RESULT_SUCCESS) {
            error = "zeDeviceGetCommandQueueGroupProperties failed";
            return;
        }
        std::vector<ze_command_queue_group_properties_t> groups(group_count);
        for (auto &group : groups) {
            group.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
        }
        if (zeDeviceGetCommandQueueGroupProperties(device, &group_count, groups.data()) !=
            ZE_RESULT_SUCCESS) {
            error = "zeDeviceGetCommandQueueGroupProperties failed";
            return;
        }
        const auto copy_group = std::find_if(groups.begin(), groups.end(), [](const auto &group) {
            return (group.flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) != 0;
        });
        if (copy_group == groups.end()) {
            error = "device has no command queue group that supports copies";
            return;
        }
        copyOrdinal = static_cast<uint32_t>(copy_group - groups.begin());

        ze_context_desc_t desc{ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
        if (zeContextCreate(driver, &desc, &context) != ZE_RESULT_SUCCESS) {
            context = nullptr;
            error = "zeContextCreate failed";
            return;
        }

        // A device can enumerate yet fail to allocate, so probe one allocation up front.
        void *probe = nullptr;
        if (allocExportable(&probe, 1) != ZE_RESULT_SUCCESS) {
            error = "exportable allocation probe failed";
            return;
        }
        zeMemFree(context, probe);
    }

    ~zeTestDevice() {
        if (context != nullptr) {
            zeContextDestroy(context);
        }
    }

    zeTestDevice(const zeTestDevice &) = delete;
    zeTestDevice &
    operator=(const zeTestDevice &) = delete;

    [[nodiscard]] bool
    usable() const {
        return error == nullptr;
    }

    // Allocate device memory that can be exported as a DMA-BUF. The NIC accesses VRAM
    // through its DMA-BUF fd. On BMG with a single device visible, the fd of a plain
    // allocation does not map the allocation's memory, so RDMA through it misses the
    // buffer, yet completes without an error.
    [[nodiscard]] ze_result_t
    allocExportable(void **ptr, size_t size) const {
        ze_external_memory_export_desc_t export_desc{ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC,
                                                     nullptr,
                                                     ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF};
        ze_device_mem_alloc_desc_t desc{
            ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, &export_desc, 0, 0};
        return zeMemAllocDevice(context, &desc, size, 0, device, ptr);
    }

    // Copy between host and device memory in either direction, returning once done.
    [[nodiscard]] ze_result_t
    copy(void *dst, const void *src, size_t size) const {
        ze_command_queue_desc_t desc{};
        desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
        desc.ordinal = copyOrdinal;
        desc.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;

        ze_command_list_handle_t list = nullptr;
        ze_result_t result = zeCommandListCreateImmediate(context, device, &desc, &list);
        if (result != ZE_RESULT_SUCCESS) {
            return result;
        }

        result = zeCommandListAppendMemoryCopy(list, dst, src, size, nullptr, 0, nullptr);
        if (result == ZE_RESULT_SUCCESS) {
            result = zeCommandListHostSynchronize(list, UINT64_MAX);
        }
        zeCommandListDestroy(list);
        return result;
    }
};

inline const zeTestDevice &
zeGetTestDevice() {
    static const zeTestDevice dev;
    return dev;
}

inline const zeTestDevice &
zeGetUsableTestDevice(const char *message) {
    const zeTestDevice &dev = zeGetTestDevice();
    if (!dev.usable()) {
        std::cerr << message << " (" << dev.error << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
    return dev;
}
#endif

inline void
gpuSetDevice(int device, const char *message) {
#if defined(HAVE_CUDA)
    checkCudaError(cudaSetDevice(device), message);
#elif defined(HAVE_ROCM)
    checkHipError(hipSetDevice(device), message);
#elif defined(HAVE_ZE)
    // Level Zero has no active device; the tests use only the first one.
    zeGetUsableTestDevice(message);
    if (device != 0) {
        std::cerr << message << " (only device 0 is supported with Level Zero)" << std::endl;
        exit(EXIT_FAILURE);
    }
#else
    (void)device;
    (void)message;
#endif
}

inline void
gpuMalloc(void **ptr, size_t size, const char *message) {
#if defined(HAVE_CUDA)
    checkCudaError(cudaMalloc(ptr, size), message);
#elif defined(HAVE_ROCM)
    checkHipError(hipMalloc(ptr, size), message);
#elif defined(HAVE_ZE)
    checkZeError(zeGetUsableTestDevice(message).allocExportable(ptr, size), message);
#else
    (void)ptr;
    (void)size;
    (void)message;
#endif
}

inline void
gpuFree(void *ptr, const char *message) {
#if defined(HAVE_CUDA)
    checkCudaError(cudaFree(ptr), message);
#elif defined(HAVE_ROCM)
    checkHipError(hipFree(ptr), message);
#elif defined(HAVE_ZE)
    checkZeError(zeMemFree(zeGetUsableTestDevice(message).context, ptr), message);
#else
    (void)ptr;
    (void)message;
#endif
}

// Copy between host and device memory; the direction is inferred from the pointers.
inline void
gpuMemcpy(void *dst, const void *src, size_t size, const char *message) {
#if defined(HAVE_CUDA)
    checkCudaError(cudaMemcpy(dst, src, size, cudaMemcpyDefault), message);
#elif defined(HAVE_ROCM)
    checkHipError(hipMemcpy(dst, src, size, hipMemcpyDefault), message);
#elif defined(HAVE_ZE)
    checkZeError(zeGetUsableTestDevice(message).copy(dst, src, size), message);
#else
    (void)dst;
    (void)src;
    (void)size;
    (void)message;
#endif
}

inline void
gpuGetDeviceCount(int *count, const char *message) {
#if defined(HAVE_CUDA)
    cudaError_t result = cudaGetDeviceCount(count);
    if (result != cudaSuccess) {
        std::cout << message << " (" << cudaGetErrorString(result) << "), setting count to 0"
                  << std::endl;
        *count = 0;
    }
#elif defined(HAVE_ROCM)
    hipError_t result = hipGetDeviceCount(count);
    if (result != hipSuccess) {
        std::cout << message << " (" << hipGetErrorString(result) << "), setting count to 0"
                  << std::endl;
        *count = 0;
    }
#elif defined(HAVE_ZE)
    // Report one device at most, since the tests use only the first one.
    const zeTestDevice &dev = zeGetTestDevice();
    if (dev.usable()) {
        *count = 1;
    } else {
        std::cout << message << " (" << dev.error << "), setting count to 0" << std::endl;
        *count = 0;
    }
#else
    std::cout << message << " (GPU support not compiled in), setting count to 0" << std::endl;
    *count = 0;
#endif
}

#endif // NIXL_TEST_GTEST_GPU_UTILS_H
