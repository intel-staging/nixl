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
 * nfi_hmem vendor: NVIDIA CUDA, through the driver API. Compiles to nothing without
 * NIXL_HAVE_CUDA_DRIVER_API.
 */

#include "nfi_hmem_vendor.h"

/****************************************
 * CUDA
 *****************************************/

#ifdef NIXL_HAVE_CUDA_DRIVER_API

#include <cuda.h>

#include <atomic>
#include <cerrno>
#include <climits>

bool
nfi_cuda_init(void) {
    if (cuInit(0) != CUDA_SUCCESS) {
        nfi_log(NFI_HMEM_LOG_DEBUG, "cuInit failed, CUDA detection disabled");
        return false;
    }
    return true;
}

static void
nfi_cuda_fill_bus(struct nfi_hmem_info *info, CUdevice dev) {
    int domain = 0, bus = 0, slot = 0;

    if (cuDeviceGetAttribute(&domain, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, dev) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&bus, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, dev) != CUDA_SUCCESS ||
        cuDeviceGetAttribute(&slot, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, dev) != CUDA_SUCCESS) {
        return;
    }

    info->bus_kind = NFI_HMEM_BUS_ACCELERATOR;
    info->bus.domain = static_cast<uint32_t>(domain);
    info->bus.bus = static_cast<uint8_t>(bus);
    info->bus.slot = static_cast<uint8_t>(slot);
    /* CUDA exposes no function number; a GPU is function 0. */
    info->bus.func = 0;
}

bool
nfi_cuda_query(const void *addr, struct nfi_hmem_info *info) {
    CUmemorytype mem_type = CU_MEMORYTYPE_HOST;
    CUdevice dev = 0;
    CUcontext ctx = nullptr;
    CUpointer_attribute attrs[3] = {CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                    CU_POINTER_ATTRIBUTE_CONTEXT};
    void *data[3] = {&mem_type, &dev, &ctx};

    /* Works without a current context: UVA lets the driver attribute any allocation. */
    if (cuPointerGetAttributes(3, attrs, data, (CUdeviceptr)addr) != CUDA_SUCCESS ||
        mem_type != CU_MEMORYTYPE_DEVICE) {
        return false;
    }

    info->iface = FI_HMEM_CUDA;
    info->device = static_cast<uint64_t>(dev);
    /* The allocating context: the primary one for the runtime API, or the application's own
     * (cuCtxCreate, green contexts). Null for virtual-memory (cuMemCreate) allocations. */
    info->context = reinterpret_cast<uint64_t>(ctx);
    nfi_cuda_fill_bus(info, dev);
    return true;
}

size_t
nfi_cuda_device_bdfs(struct nfi_hmem_bus_id *out, size_t max) {
    int count = 0;
    size_t found = 0;

    if (cuDeviceGetCount(&count) != CUDA_SUCCESS) {
        return 0;
    }
    for (int i = 0; i < count; i++) {
        CUdevice dev;
        if (cuDeviceGet(&dev, i) != CUDA_SUCCESS) {
            continue;
        }
        struct nfi_hmem_info info = {};
        nfi_cuda_fill_bus(&info, dev);
        if (info.bus_kind == NFI_HMEM_BUS_NONE) {
            continue;
        }
        if (out && found < max) {
            out[found] = info.bus;
        }
        found++;
    }
    return found;
}

/*
 * Primary context per CUDA device ordinal, retained on first use and kept for the process so a
 * scope push is a single cuCtxPushCurrent. Ordinals past the table are retained and released per
 * scope. If a push fails because the application reset the device, the context is retained again
 * and the push retried once. The replaced entry's retain is not released: whether a reset leaves
 * it counted is up to the driver, and releasing it could destroy the context just retained. At
 * most one reference per reset is held until exit.
 */
#define NFI_CUDA_MAX_CACHED_DEVICES 64
static std::atomic<CUcontext> nfi_cuda_primary[NFI_CUDA_MAX_CACHED_DEVICES];

/*
 * Retained primary context for @p dev, or nullptr on failure. *@p cached says whether the retain
 * belongs to the cache. @p stale, when not nullptr, is a cached context to replace.
 */
static CUcontext
nfi_cuda_primary_ctx(CUdevice dev, bool *cached, CUcontext stale) {
    const bool cacheable = dev >= 0 && dev < NFI_CUDA_MAX_CACHED_DEVICES;
    *cached = cacheable;

    if (cacheable && !stale) {
        CUcontext ctx = nfi_cuda_primary[dev].load(std::memory_order_acquire);
        if (ctx) {
            return ctx;
        }
    }

    CUcontext primary = nullptr;
    if (cuDevicePrimaryCtxRetain(&primary, dev) != CUDA_SUCCESS) {
        nfi_log(NFI_HMEM_LOG_ERROR,
                "cuDevicePrimaryCtxRetain failed for CUDA device %d",
                static_cast<int>(dev));
        return nullptr;
    }
    if (!cacheable) {
        return primary;
    }

    /* A thread that loses the race drops its retain and uses the winner's, the same context. */
    CUcontext expected = stale;
    if (!nfi_cuda_primary[dev].compare_exchange_strong(
            expected, primary, std::memory_order_acq_rel)) {
        (void)cuDevicePrimaryCtxRelease(dev);
        return expected;
    }
    return primary;
}

/* True only when the driver reports @p dev's primary context inactive. */
static bool
nfi_cuda_primary_ctx_inactive(CUdevice dev) {
    unsigned int flags = 0;
    int active = 0;
    return cuDevicePrimaryCtxGetState(dev, &flags, &active) == CUDA_SUCCESS && !active;
}

/* *saved holds the device, with the top bit set when pop has no retain to release (the cache
 * owns it, or the context is the allocation's own). */
#define NFI_CUDA_SAVED_CACHED (1ULL << 63)

int
nfi_cuda_scope_push(uint64_t device, uint64_t context, uint64_t *saved) {
    /* An ordinal, as nfi_cuda_query() reports it; one past INT_MAX cannot be a CUdevice. */
    if (device > static_cast<uint64_t>(INT_MAX)) {
        return -EINVAL;
    }
    CUdevice dev = static_cast<CUdevice>(device);

    /* The allocation's own context, which the application keeps alive while the memory is
     * registered: pushed as is, with no retain to drop on pop. */
    if (context != 0) {
        if (cuCtxPushCurrent(reinterpret_cast<CUcontext>(context)) != CUDA_SUCCESS) {
            nfi_log(NFI_HMEM_LOG_ERROR,
                    "cuCtxPushCurrent failed for the owning context of CUDA device %d",
                    static_cast<int>(dev));
            return -EIO;
        }
        *saved = device | NFI_CUDA_SAVED_CACHED;
        return 0;
    }
    bool cached = false;
    CUcontext primary = nfi_cuda_primary_ctx(dev, &cached, nullptr);
    if (!primary) {
        return -EIO;
    }

    CUresult ret = cuCtxPushCurrent(primary);
    if (ret != CUDA_SUCCESS && cached && nfi_cuda_primary_ctx_inactive(dev)) {
        nfi_log(NFI_HMEM_LOG_INFO,
                "primary context of CUDA device %d is no longer active, retaining it again",
                static_cast<int>(dev));
        primary = nfi_cuda_primary_ctx(dev, &cached, primary);
        ret = primary ? cuCtxPushCurrent(primary) : CUDA_ERROR_INVALID_CONTEXT;
    }
    if (ret != CUDA_SUCCESS) {
        if (primary && !cached) {
            (void)cuDevicePrimaryCtxRelease(dev);
        }
        nfi_log(NFI_HMEM_LOG_ERROR,
                "cuCtxPushCurrent failed for CUDA device %d",
                static_cast<int>(dev));
        return -EIO;
    }

    *saved = device | (cached ? NFI_CUDA_SAVED_CACHED : 0);
    return 0;
}

void
nfi_cuda_scope_pop(uint64_t saved) {
    CUcontext popped = nullptr;

    if (cuCtxPopCurrent(&popped) != CUDA_SUCCESS) {
        nfi_log(NFI_HMEM_LOG_WARN, "cuCtxPopCurrent failed");
    }
    if (saved & NFI_CUDA_SAVED_CACHED) {
        return;
    }
    if (cuDevicePrimaryCtxRelease(static_cast<CUdevice>(saved)) != CUDA_SUCCESS) {
        nfi_log(NFI_HMEM_LOG_WARN,
                "cuDevicePrimaryCtxRelease failed for CUDA device %d",
                static_cast<int>(saved));
    }
}

#endif /* NIXL_HAVE_CUDA_DRIVER_API */
