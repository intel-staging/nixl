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
 * Vendor table for nfi_hmem.h. Each runtime is reached through dlopen, so only its SDK headers are
 * a build-time dependency, and its section compiles out without them.
 */

#include "nfi_hmem.h"

#include <dlfcn.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#ifdef NIXL_HAVE_CUDA_DRIVER_API
#include <cuda.h>
#endif

#ifdef HAVE_ZE
#include <level_zero/ze_api.h>
#endif

#ifdef NIXL_HAVE_ROCM
#include <hip/hip_runtime_api.h>
#endif

/****************************************
 * Diagnostics
 *****************************************/

static nfi_hmem_log_fn nfi_log_fn;

void
nfi_hmem_set_log_fn(nfi_hmem_log_fn fn) {
    nfi_log_fn = fn;
}

__attribute__((format(printf, 2, 3))) static void
nfi_log(enum nfi_hmem_log_level level, const char *fmt, ...) {
    /* Read once: a consumer may install or clear the sink concurrently. */
    const nfi_hmem_log_fn fn = nfi_log_fn;
    if (fn == nullptr) {
        return;
    }

    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    fn(level, buf);
}

/****************************************
 * Vendor table shape
 *****************************************/

struct nfi_vendor {
    enum fi_hmem_iface iface;

    /* Runs once per process; false makes the vendor unavailable. nullptr without the SDK. */
    bool (*init)(void);

    /* Returns false for a pointer the runtime does not own. nullptr without the SDK. */
    bool (*query)(const void *addr, struct nfi_hmem_info *info);

    /* Enumerates accelerator PCI addresses; nullptr for a vendor that cannot. */
    size_t (*device_bdfs)(struct nfi_hmem_bus_id *out, size_t max);

    /* Thread-current context management; nullptr for a runtime without such state. */
    int (*scope_push)(uint64_t device, uint64_t *saved);
    void (*scope_pop)(uint64_t saved);
};

/****************************************
 * Neuron
 *****************************************/

static int (*nfi_nrt_get_attached_efa_bdf)(const void *va, char *efa_bdf, size_t *len);

static bool
nfi_neuron_init(void) {
    void *handle = dlopen("libnrt.so.1", RTLD_NOW);
    if (!handle) {
        nfi_log(NFI_HMEM_LOG_DEBUG, "libnrt.so.1 not available, Neuron detection disabled");
        return false;
    }

    nfi_nrt_get_attached_efa_bdf =
        (int (*)(const void *, char *, size_t *))dlsym(handle, "nrt_get_attached_efa_bdf");
    if (!nfi_nrt_get_attached_efa_bdf) {
        nfi_log(NFI_HMEM_LOG_WARN, "could not resolve libnrt symbol nrt_get_attached_efa_bdf");
        return false;
    }
    return true;
}

static bool
nfi_neuron_query(const void *addr, struct nfi_hmem_info *info) {
    char buf[] = "0000:00:00.0";
    size_t buflen = sizeof(buf);
    unsigned domain, bus, slot, func;

    if (nfi_nrt_get_attached_efa_bdf(addr, buf, &buflen) != 0) {
        return false;
    }

    info->iface = FI_HMEM_NEURON;
    /* libfabric requires device.neuron to be set but the EFA provider does not read it. */
    info->device = static_cast<uint64_t>(-1);

    /* libnrt reports the address of the EFA NIC attached to the accelerator. */
    if (sscanf(buf, "%x:%x:%x.%x", &domain, &bus, &slot, &func) == 4) {
        info->bus_kind = NFI_HMEM_BUS_NIC;
        info->bus.domain = static_cast<uint16_t>(domain);
        info->bus.bus = static_cast<uint8_t>(bus);
        info->bus.slot = static_cast<uint8_t>(slot);
        info->bus.func = static_cast<uint8_t>(func);
    }
    return true;
}

/****************************************
 * Level Zero
 *****************************************/

#ifdef HAVE_ZE

/* Entry points resolved from libze_loader with dlsym. */
struct nfi_ze_ops {
    ze_result_t (*init)(ze_init_flags_t);
    ze_result_t (*driver_get)(uint32_t *, ze_driver_handle_t *);
    ze_result_t (*device_get)(ze_driver_handle_t, uint32_t *, ze_device_handle_t *);
    ze_result_t (*device_get_sub)(ze_device_handle_t, uint32_t *, ze_device_handle_t *);
    ze_result_t (*context_create)(ze_driver_handle_t,
                                  const ze_context_desc_t *,
                                  ze_context_handle_t *);
    ze_result_t (*device_pci_props)(ze_device_handle_t, ze_pci_ext_properties_t *);
    ze_result_t (*device_props)(ze_device_handle_t, ze_device_properties_t *);
    ze_result_t (*mem_alloc_props)(ze_context_handle_t,
                                   const void *,
                                   ze_memory_allocation_properties_t *,
                                   ze_device_handle_t *);
    /* Optional: without it the pooling check in nfi_ze_check_pooled() is skipped. */
    ze_result_t (*mem_address_range)(ze_context_handle_t, const void *, void **, size_t *);
};

/* One Level Zero device. Kept in zeDeviceGet() order, so the index is libfabric's ordinal. */
struct nfi_ze_device {
    ze_device_handle_t handle = nullptr;
    struct nfi_hmem_bus_id bus = {};
    /* Whether bus was readable; an all-zero address is valid. */
    bool has_bus = false;
    /* Not ZE_DEVICE_PROPERTY_FLAG_INTEGRATED. Only discrete GPUs are reported as accelerators. */
    bool discrete = false;
};

static struct nfi_ze_ops nfi_ze;
static ze_context_handle_t nfi_ze_context;
static std::vector<nfi_ze_device> nfi_ze_devices;

static bool
nfi_ze_load(void *handle) {
    struct {
        const char *name;
        void **slot;
    } syms[] = {
        {"zeInit", (void **)&nfi_ze.init},
        {"zeDriverGet", (void **)&nfi_ze.driver_get},
        {"zeDeviceGet", (void **)&nfi_ze.device_get},
        {"zeDeviceGetSubDevices", (void **)&nfi_ze.device_get_sub},
        {"zeContextCreate", (void **)&nfi_ze.context_create},
        {"zeDevicePciGetPropertiesExt", (void **)&nfi_ze.device_pci_props},
        {"zeDeviceGetProperties", (void **)&nfi_ze.device_props},
        {"zeMemGetAllocProperties", (void **)&nfi_ze.mem_alloc_props},
    };

    for (const auto &sym : syms) {
        *sym.slot = dlsym(handle, sym.name);
        if (*sym.slot == nullptr) {
            nfi_log(NFI_HMEM_LOG_WARN, "could not resolve Level Zero symbol %s", sym.name);
            return false;
        }
    }
    nfi_ze.mem_address_range =
        reinterpret_cast<decltype(nfi_ze.mem_address_range)>(dlsym(handle, "zeMemGetAddressRange"));
    return true;
}

static bool
nfi_ze_init(void) {
    uint32_t driver_count = 0;
    uint32_t device_count = 0;

    void *handle = dlopen("libze_loader.so.1", RTLD_NOW);
    if (handle == nullptr) {
        nfi_log(NFI_HMEM_LOG_DEBUG,
                "libze_loader.so.1 not available, Level Zero detection disabled");
        return false;
    }

    if (!nfi_ze_load(handle)) {
        return false;
    }

    /*
     * GPU_ONLY, as libfabric's hmem_ze.c does, so driver 0 here is the same driver libfabric uses
     * (flags=0 would also enumerate e.g. the NPU driver).
     */
    if (nfi_ze.init(ZE_INIT_FLAG_GPU_ONLY) != ZE_RESULT_SUCCESS) {
        nfi_log(NFI_HMEM_LOG_DEBUG, "zeInit failed, Level Zero detection disabled");
        return false;
    }

    if (nfi_ze.driver_get(&driver_count, nullptr) != ZE_RESULT_SUCCESS || driver_count == 0) {
        nfi_log(NFI_HMEM_LOG_DEBUG, "no Level Zero drivers found");
        return false;
    }
    std::vector<ze_driver_handle_t> drivers(driver_count);
    if (nfi_ze.driver_get(&driver_count, drivers.data()) != ZE_RESULT_SUCCESS) {
        nfi_log(NFI_HMEM_LOG_WARN, "zeDriverGet failed to retrieve driver handles");
        return false;
    }

    /* Driver 0 only: libfabric's Level Zero support is single-driver. */
    if (driver_count > 1) {
        nfi_log(NFI_HMEM_LOG_DEBUG,
                "found %u Level Zero drivers, using driver 0 only (libfabric is single-driver)",
                static_cast<unsigned>(driver_count));
    }
    ze_driver_handle_t driver = drivers[0];

    if (nfi_ze.device_get(driver, &device_count, nullptr) != ZE_RESULT_SUCCESS ||
        device_count == 0) {
        nfi_log(NFI_HMEM_LOG_DEBUG, "no Level Zero devices on driver 0");
        return false;
    }
    std::vector<ze_device_handle_t> handles(device_count);
    if (nfi_ze.device_get(driver, &device_count, handles.data()) != ZE_RESULT_SUCCESS) {
        nfi_log(NFI_HMEM_LOG_WARN, "zeDeviceGet failed to retrieve device handles");
        return false;
    }
    ze_context_desc_t ctx_desc = {};
    ctx_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    if (nfi_ze.context_create(driver, &ctx_desc, &nfi_ze_context) != ZE_RESULT_SUCCESS) {
        nfi_log(NFI_HMEM_LOG_WARN, "zeContextCreate failed, Level Zero detection disabled");
        nfi_ze_context = nullptr;
        return false;
    }

    nfi_ze_devices.reserve(handles.size());
    for (size_t i = 0; i < handles.size(); i++) {
        nfi_ze_device dev;
        dev.handle = handles[i];

        ze_pci_ext_properties_t pci = {};
        pci.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
        if (nfi_ze.device_pci_props(dev.handle, &pci) == ZE_RESULT_SUCCESS) {
            dev.bus.domain = static_cast<uint16_t>(pci.address.domain);
            dev.bus.bus = static_cast<uint8_t>(pci.address.bus);
            dev.bus.slot = static_cast<uint8_t>(pci.address.device);
            dev.bus.func = static_cast<uint8_t>(pci.address.function);
            dev.has_bus = true;
        } else {
            nfi_log(NFI_HMEM_LOG_DEBUG,
                    "zeDevicePciGetPropertiesExt failed for Level Zero device %u",
                    static_cast<unsigned>(i));
        }

        /*
         * Integrated vs discrete comes from the driver: both report PCI class 0x0300, and
         * compute-only parts report no display class.
         */
        ze_device_properties_t props = {};
        props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
        if (nfi_ze.device_props(dev.handle, &props) == ZE_RESULT_SUCCESS) {
            dev.discrete = (props.type == ZE_DEVICE_TYPE_GPU) &&
                ((props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) == 0);
        } else {
            nfi_log(NFI_HMEM_LOG_DEBUG,
                    "zeDeviceGetProperties failed for Level Zero device %u, not treating it as "
                    "discrete",
                    static_cast<unsigned>(i));
        }

        nfi_ze_devices.push_back(dev);
    }

    nfi_log(NFI_HMEM_LOG_INFO,
            "Level Zero detection enabled with %u device(s) on driver 0",
            static_cast<unsigned>(nfi_ze_devices.size()));
    return true;
}

/*
 * Index of the root device owning @p handle, or -1. zeMemGetAllocProperties() can return a
 * sub-device (tile) handle, while zeDeviceGet() lists root devices only.
 */
static int
nfi_ze_device_index(ze_device_handle_t handle) {
    for (size_t i = 0; i < nfi_ze_devices.size(); i++) {
        if (nfi_ze_devices[i].handle == handle) {
            return static_cast<int>(i);
        }
    }

    for (size_t i = 0; i < nfi_ze_devices.size(); i++) {
        uint32_t sub_count = 0;
        if (nfi_ze.device_get_sub(nfi_ze_devices[i].handle, &sub_count, nullptr) !=
                ZE_RESULT_SUCCESS ||
            sub_count == 0) {
            continue;
        }

        std::vector<ze_device_handle_t> subs(sub_count);
        if (nfi_ze.device_get_sub(nfi_ze_devices[i].handle, &sub_count, subs.data()) !=
            ZE_RESULT_SUCCESS) {
            continue;
        }

        for (uint32_t sub = 0; sub < sub_count; sub++) {
            if (subs[sub] == handle) {
                nfi_log(NFI_HMEM_LOG_DEBUG,
                        "resolved Level Zero sub-device %u to root device %u",
                        static_cast<unsigned>(sub),
                        static_cast<unsigned>(i));
                return static_cast<int>(i);
            }
        }
    }

    return -1;
}

/*
 * Detects an allocation that is a slice of a pooled buffer object. Level Zero pools small
 * allocations made without ze_external_memory_export_desc_t, and libfabric would register them at
 * the wrong dma-buf offset. Exports the dma-buf as libfabric does and compares its size with the
 * allocation's, rounded up to the 64 KiB minimum buffer object size.
 */
static void
nfi_ze_check_pooled(const void *addr, struct nfi_hmem_info *info) {
    constexpr uint64_t min_bo_size = 64 * 1024;
    void *base = nullptr;
    size_t size = 0;

    if (nfi_ze.mem_address_range == nullptr ||
        nfi_ze.mem_address_range(nfi_ze_context, addr, &base, &size) != ZE_RESULT_SUCCESS ||
        size == 0) {
        return;
    }

    ze_external_memory_export_fd_t export_fd = {};
    export_fd.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD;
    export_fd.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF;
    export_fd.fd = -1;
    ze_memory_allocation_properties_t props = {};
    props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;
    props.pNext = &export_fd;
    ze_device_handle_t device = nullptr;
    if (nfi_ze.mem_alloc_props(nfi_ze_context, addr, &props, &device) != ZE_RESULT_SUCCESS ||
        export_fd.fd < 0) {
        /* Not exportable: nothing to compare. */
        return;
    }
    /*
     * Not closed: the driver returns the same fd to every export of a buffer object, libfabric's
     * included, and closing it breaks the registration that follows. The offset is restored.
     */
    const off_t dmabuf_size = lseek(export_fd.fd, 0, SEEK_END);
    lseek(export_fd.fd, 0, SEEK_SET);
    if (dmabuf_size <= 0) {
        return;
    }

    const uint64_t rounded =
        (static_cast<uint64_t>(size) + min_bo_size - 1) / min_bo_size * min_bo_size;
    if (static_cast<uint64_t>(dmabuf_size) > rounded) {
        info->pooled_dmabuf_size = static_cast<uint64_t>(dmabuf_size);
        info->alloc_size = static_cast<uint64_t>(size);
        nfi_log(NFI_HMEM_LOG_DEBUG,
                "Level Zero allocation of %zu bytes at %p is part of a %lld-byte pooled buffer",
                size,
                base,
                static_cast<long long>(dmabuf_size));
    }
}

static bool
nfi_ze_query(const void *addr, struct nfi_hmem_info *info) {
    ze_memory_allocation_properties_t props = {};
    ze_device_handle_t device = nullptr;
    int index = 0;

    props.stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES;

    if (nfi_ze.mem_alloc_props(nfi_ze_context, addr, &props, &device) != ZE_RESULT_SUCCESS) {
        return false;
    }

    /* As in libfabric's ze_hmem_is_addr_valid(): DEVICE, SHARED and HOST memory are all ZE. */
    if (props.type == ZE_MEMORY_TYPE_UNKNOWN) {
        return false;
    }

    if (device == nullptr) {
        /* ZE_MEMORY_TYPE_HOST belongs to no device, so there is no ordinal or address. */
        nfi_log(NFI_HMEM_LOG_DEBUG,
                "Level Zero allocation %p has no owning device (type %d), reporting ordinal 0",
                addr,
                static_cast<int>(props.type));
    } else {
        index = nfi_ze_device_index(device);
        if (index < 0) {
            /* Still claimed: a wrong ordinal is recoverable, a wrong iface is not. */
            nfi_log(NFI_HMEM_LOG_WARN,
                    "Level Zero device for %p is not in the driver 0 device list, using ordinal 0",
                    addr);
            index = 0;
        } else if (nfi_ze_devices[index].has_bus) {
            info->bus_kind = NFI_HMEM_BUS_ACCELERATOR;
            info->bus = nfi_ze_devices[index].bus;
        }
    }

    info->iface = FI_HMEM_ZE;
    info->device = static_cast<uint64_t>(fi_hmem_ze_device(0, index));
    nfi_ze_check_pooled(addr, info);
    return true;
}

static size_t
nfi_ze_device_bdfs(struct nfi_hmem_bus_id *out, size_t max) {
    size_t found = 0;

    for (const nfi_ze_device &dev : nfi_ze_devices) {
        if (!dev.discrete || !dev.has_bus) {
            continue;
        }
        if (out != nullptr && found < max) {
            out[found] = dev.bus;
        }
        found++;
    }

    return found;
}

#endif /* HAVE_ZE */

/****************************************
 * CUDA
 *****************************************/

#ifdef NIXL_HAVE_CUDA_DRIVER_API

static bool
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
    info->bus.domain = static_cast<uint16_t>(domain);
    info->bus.bus = static_cast<uint8_t>(bus);
    info->bus.slot = static_cast<uint8_t>(slot);
    /* CUDA exposes no function number; a GPU is function 0. */
    info->bus.func = 0;
}

static bool
nfi_cuda_query(const void *addr, struct nfi_hmem_info *info) {
    CUmemorytype mem_type = CU_MEMORYTYPE_HOST;
    CUdevice dev = 0;
    CUpointer_attribute attrs[2] = {CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL};
    void *data[2] = {&mem_type, &dev};

    /* Works without a current context: UVA lets the driver attribute any allocation. */
    if (cuPointerGetAttributes(2, attrs, data, (CUdeviceptr)addr) != CUDA_SUCCESS ||
        mem_type != CU_MEMORYTYPE_DEVICE) {
        return false;
    }

    info->iface = FI_HMEM_CUDA;
    info->device = static_cast<uint64_t>(dev);
    nfi_cuda_fill_bus(info, dev);
    return true;
}

static size_t
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
 * and the push retried once.
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

/* *saved holds the device, with the top bit set when its retain belongs to the cache. */
#define NFI_CUDA_SAVED_CACHED (1ULL << 63)

static int
nfi_cuda_scope_push(uint64_t device, uint64_t *saved) {
    CUdevice dev = (CUdevice)device;
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

static void
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

/****************************************
 * AMD ROCr (HIP)
 *****************************************/

#ifdef NIXL_HAVE_ROCM

/* dlopen'd entry points of libamdhip64. ROCr keeps no thread-current state, so no scope hooks. */
struct nfi_rocr_ops {
    hipError_t (*get_device_count)(int *);
    hipError_t (*pointer_get_attributes)(hipPointerAttribute_t *, const void *);
    hipError_t (*device_get_pci_bus_id)(char *, int, int);
};

static struct nfi_rocr_ops nfi_rocr;

static bool
nfi_rocr_init(void) {
    void *handle = dlopen("libamdhip64.so", RTLD_NOW);
    if (handle == nullptr) {
        nfi_log(NFI_HMEM_LOG_DEBUG, "libamdhip64.so not available, ROCr detection disabled");
        return false;
    }

    struct {
        const char *name;
        void **slot;
    } syms[] = {
        {"hipGetDeviceCount", (void **)&nfi_rocr.get_device_count},
        {"hipPointerGetAttributes", (void **)&nfi_rocr.pointer_get_attributes},
        {"hipDeviceGetPCIBusId", (void **)&nfi_rocr.device_get_pci_bus_id},
    };

    for (const auto &sym : syms) {
        *sym.slot = dlsym(handle, sym.name);
        if (*sym.slot == nullptr) {
            nfi_log(NFI_HMEM_LOG_WARN, "could not resolve HIP symbol %s", sym.name);
            return false;
        }
    }

    int count = 0;
    if (nfi_rocr.get_device_count(&count) != hipSuccess || count <= 0) {
        nfi_log(NFI_HMEM_LOG_DEBUG, "no HIP devices found, ROCr detection disabled");
        return false;
    }
    nfi_log(NFI_HMEM_LOG_INFO, "ROCr detection enabled with %d device(s)", count);
    return true;
}

static void
nfi_rocr_fill_bus(struct nfi_hmem_info *info, int dev) {
    char buf[32];
    unsigned domain, bus, slot, func;

    if (nfi_rocr.device_get_pci_bus_id(buf, sizeof(buf), dev) != hipSuccess) {
        return;
    }
    if (sscanf(buf, "%x:%x:%x.%x", &domain, &bus, &slot, &func) == 4) {
        info->bus_kind = NFI_HMEM_BUS_ACCELERATOR;
        info->bus.domain = static_cast<uint16_t>(domain);
        info->bus.bus = static_cast<uint8_t>(bus);
        info->bus.slot = static_cast<uint8_t>(slot);
        info->bus.func = static_cast<uint8_t>(func);
    }
}

static bool
nfi_rocr_query(const void *addr, struct nfi_hmem_info *info) {
    hipPointerAttribute_t attr = {};

    if (nfi_rocr.pointer_get_attributes(&attr, addr) != hipSuccess) {
        return false;
    }
#if HIP_VERSION_MAJOR >= 6
    const hipMemoryType mem_type = attr.type;
#else
    const hipMemoryType mem_type = attr.memoryType;
#endif
    if (mem_type != hipMemoryTypeDevice) {
        return false;
    }

    info->iface = FI_HMEM_ROCR;
    info->device = static_cast<uint64_t>(attr.device);
    nfi_rocr_fill_bus(info, attr.device);
    return true;
}

static size_t
nfi_rocr_device_bdfs(struct nfi_hmem_bus_id *out, size_t max) {
    int count = 0;
    size_t found = 0;

    if (nfi_rocr.get_device_count(&count) != hipSuccess) {
        return 0;
    }
    for (int i = 0; i < count; i++) {
        struct nfi_hmem_info info = {};
        nfi_rocr_fill_bus(&info, i);
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

#endif /* NIXL_HAVE_ROCM */

/****************************************
 * Table
 *****************************************/

/*
 * Probe order: descending fi_hmem_iface, as libfabric's ofi_get_hmem_iface() iterates, so this file
 * and providers that call it (cxi, rxm, opx) attribute a pointer to the same runtime.
 */
static const struct nfi_vendor nfi_vendors[] = {
    {FI_HMEM_NEURON, nfi_neuron_init, nfi_neuron_query, nullptr, nullptr, nullptr},
#ifdef HAVE_ZE
    {FI_HMEM_ZE, nfi_ze_init, nfi_ze_query, nfi_ze_device_bdfs, nullptr, nullptr},
#else
    {FI_HMEM_ZE, nullptr, nullptr, nullptr, nullptr, nullptr},
#endif
#ifdef NIXL_HAVE_ROCM
    {FI_HMEM_ROCR, nfi_rocr_init, nfi_rocr_query, nfi_rocr_device_bdfs, nullptr, nullptr},
#else
    {FI_HMEM_ROCR, nullptr, nullptr, nullptr, nullptr, nullptr},
#endif
#ifdef NIXL_HAVE_CUDA_DRIVER_API
    {FI_HMEM_CUDA,
     nfi_cuda_init,
     nfi_cuda_query,
     nfi_cuda_device_bdfs,
     nfi_cuda_scope_push,
     nfi_cuda_scope_pop},
#else
    {FI_HMEM_CUDA, nullptr, nullptr, nullptr, nullptr, nullptr},
#endif
};

#define NFI_NUM_VENDORS (sizeof(nfi_vendors) / sizeof(nfi_vendors[0]))

/*
 * Per-vendor init state, parallel to nfi_vendors[]. The mutex serialises init only; once settled,
 * the state is read with a single acquire load.
 */
enum nfi_vendor_state : unsigned char {
    NFI_VENDOR_UNTRIED = 0,
    NFI_VENDOR_OK,
    NFI_VENDOR_UNAVAILABLE,
};

static std::mutex nfi_vendor_lock;
static std::atomic<unsigned char> nfi_vendor_state[NFI_NUM_VENDORS];

static bool
nfi_vendor_available(size_t index) {
    if (index >= NFI_NUM_VENDORS) {
        return false;
    }

    unsigned char state = nfi_vendor_state[index].load(std::memory_order_acquire);
    if (state != NFI_VENDOR_UNTRIED) {
        return state == NFI_VENDOR_OK;
    }

    const std::lock_guard<std::mutex> lock(nfi_vendor_lock);
    state = nfi_vendor_state[index].load(std::memory_order_relaxed);
    if (state == NFI_VENDOR_UNTRIED) {
        const bool ok =
            nfi_vendors[index].init && nfi_vendors[index].query && nfi_vendors[index].init();
        state = ok ? NFI_VENDOR_OK : NFI_VENDOR_UNAVAILABLE;
        nfi_vendor_state[index].store(state, std::memory_order_release);
    }
    return state == NFI_VENDOR_OK;
}

static size_t
nfi_vendor_index(enum fi_hmem_iface iface) {
    for (size_t i = 0; i < NFI_NUM_VENDORS; i++) {
        if (nfi_vendors[i].iface == iface) {
            return i;
        }
    }
    return NFI_NUM_VENDORS;
}

/****************************************
 * Public interface
 *****************************************/

int
nfi_hmem_query_ptr(const void *addr, struct nfi_hmem_info *info) {
    if (!addr || !info) {
        return -EINVAL;
    }

    *info = {};
    info->iface = FI_HMEM_SYSTEM;

    for (size_t i = 0; i < NFI_NUM_VENDORS; i++) {
        if (!nfi_vendor_available(i)) {
            continue;
        }
        if (nfi_vendors[i].query(addr, info)) {
            nfi_log(NFI_HMEM_LOG_DEBUG,
                    "%s claims %p (device %llu)",
                    nfi_hmem_iface_name(nfi_vendors[i].iface),
                    addr,
                    (unsigned long long)info->device);
            return 0;
        }
        /* Discard partial output from a failed probe. */
        *info = {};
        info->iface = FI_HMEM_SYSTEM;
    }

    nfi_log(
        NFI_HMEM_LOG_DEBUG, "no accelerator runtime claimed %p, treating as system memory", addr);
    return 0;
}

void
nfi_hmem_fill_mr_attr(const struct nfi_hmem_info *info, struct fi_mr_attr *attr) {
    if (!info || !attr) {
        return;
    }

    attr->iface = info->iface;

    switch (info->iface) {
    case FI_HMEM_CUDA:
        attr->device.cuda = static_cast<int>(info->device);
        break;
    case FI_HMEM_ZE:
        /* Already encoded as fi_hmem_ze_device(0, index) by nfi_ze_query(). */
        attr->device.ze = static_cast<int>(info->device);
        break;
    case FI_HMEM_NEURON:
        attr->device.neuron = static_cast<int>(info->device);
        break;
    case FI_HMEM_ROCR:
        /* device.rocr exists from libfabric 2.3; older versions leave the union zeroed. */
#if FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION) >= FI_VERSION(2, 3)
        attr->device.rocr = static_cast<int>(info->device);
#endif
        break;
    case FI_HMEM_SYSTEM:
        /* libfabric expects device unset for host memory. */
        break;
    default:
        nfi_log(NFI_HMEM_LOG_WARN,
                "no fi_mr_attr::device arm for hmem iface %d, registering with device unset",
                static_cast<int>(info->iface));
        break;
    }
}

bool
nfi_hmem_available(void) {
    for (size_t i = 0; i < NFI_NUM_VENDORS; i++) {
        if (nfi_vendor_available(i)) {
            return true;
        }
    }
    return false;
}

bool
nfi_hmem_iface_available(enum fi_hmem_iface iface) {
    return nfi_vendor_available(nfi_vendor_index(iface));
}

enum fi_hmem_iface
nfi_hmem_primary_iface(void) {
    for (size_t i = 0; i < NFI_NUM_VENDORS; i++) {
        if (nfi_vendor_available(i)) {
            return nfi_vendors[i].iface;
        }
    }
    return FI_HMEM_SYSTEM;
}

const enum fi_hmem_iface *
nfi_hmem_ifaces(size_t *count) {
    static enum fi_hmem_iface ifaces[NFI_NUM_VENDORS];
    /* Concurrent callers write identical values, so no once-flag is needed. */
    for (size_t i = 0; i < NFI_NUM_VENDORS; i++) {
        ifaces[i] = nfi_vendors[i].iface;
    }

    if (count) {
        *count = NFI_NUM_VENDORS;
    }
    return ifaces;
}

/* Names from fi_tostr_r(FI_TYPE_HMEM_IFACE) without the "FI_HMEM_" prefix, cached once. */
#define NFI_IFACE_NAME_SLOTS 16
#define NFI_IFACE_NAME_MAX 32

static char nfi_iface_names[NFI_IFACE_NAME_SLOTS][NFI_IFACE_NAME_MAX];
static std::once_flag nfi_iface_names_once;

static void
nfi_init_iface_names(void) {
    static const char prefix[] = "FI_HMEM_";
    const size_t prefix_len = sizeof(prefix) - 1;
    int slot;

    for (slot = 0; slot < NFI_IFACE_NAME_SLOTS; slot++) {
        enum fi_hmem_iface iface = (enum fi_hmem_iface)slot;
        char buf[NFI_IFACE_NAME_MAX];
        const char *name = buf;

        fi_tostr_r(buf, sizeof(buf), &iface, FI_TYPE_HMEM_IFACE);
        if (!strncmp(name, prefix, prefix_len)) {
            name += prefix_len;
        }

        snprintf(nfi_iface_names[slot], NFI_IFACE_NAME_MAX, "%s", name);
    }
}

const char *
nfi_hmem_iface_name(enum fi_hmem_iface iface) {
    const int slot = static_cast<int>(iface);

    std::call_once(nfi_iface_names_once, nfi_init_iface_names);

    if (slot < 0 || slot >= NFI_IFACE_NAME_SLOTS) {
        return "Unknown";
    }
    return nfi_iface_names[slot];
}

size_t
nfi_hmem_device_bdfs(enum fi_hmem_iface iface, struct nfi_hmem_bus_id *out, size_t max) {
    size_t index = nfi_vendor_index(iface);

    if (index >= NFI_NUM_VENDORS || !nfi_vendors[index].device_bdfs) {
        return 0;
    }
    if (!nfi_vendor_available(index)) {
        return 0;
    }

    return nfi_vendors[index].device_bdfs(out, out ? max : 0);
}

bool
nfi_hmem_needs_scope(enum fi_hmem_iface iface) {
    size_t index = nfi_vendor_index(iface);

    if (index >= NFI_NUM_VENDORS || !nfi_vendors[index].scope_push) {
        return false;
    }
    return nfi_vendor_available(index);
}

int
nfi_hmem_scope_push(enum fi_hmem_iface iface, uint64_t device, struct nfi_hmem_scope *scope) {
    if (!scope) {
        return -EINVAL;
    }

    scope->iface = FI_HMEM_SYSTEM;
    scope->saved = 0;
    scope->pushed = 0;

    size_t index = nfi_vendor_index(iface);
    if (index >= NFI_NUM_VENDORS || !nfi_vendors[index].scope_push ||
        !nfi_vendor_available(index)) {
        return 0;
    }

    int ret = nfi_vendors[index].scope_push(device, &scope->saved);
    if (ret) {
        return ret;
    }

    scope->iface = iface;
    scope->pushed = 1;
    return 0;
}

void
nfi_hmem_scope_pop(struct nfi_hmem_scope *scope) {
    if (!scope || !scope->pushed) {
        return;
    }

    size_t index = nfi_vendor_index(scope->iface);
    if (index < NFI_NUM_VENDORS && nfi_vendors[index].scope_pop) {
        nfi_vendors[index].scope_pop(scope->saved);
    }
    scope->pushed = 0;
}
