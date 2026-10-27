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
 * nfi_hmem vendor: AMD ROCr, through a dlopen'd libamdhip64. Compiles to nothing without
 * NIXL_HAVE_ROCM.
 */

#include "nfi_hmem_vendor.h"


/****************************************
 * AMD ROCr (HIP)
 *****************************************/

#ifdef NIXL_HAVE_ROCM

#include <hip/hip_runtime_api.h>

#include <dlfcn.h>

#include <cstdio>

/* dlopen'd entry points of libamdhip64. ROCr keeps no thread-current state, so no scope hooks. */
struct nfi_rocr_ops {
    hipError_t (*get_device_count)(int *);
    hipError_t (*pointer_get_attributes)(hipPointerAttribute_t *, const void *);
    hipError_t (*device_get_pci_bus_id)(char *, int, int);
};

static struct nfi_rocr_ops nfi_rocr;

bool
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

bool
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

size_t
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
