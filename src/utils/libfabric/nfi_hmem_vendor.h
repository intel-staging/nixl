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
 * Internal to nfi_hmem: the shared diagnostic hook and the entry points of the vendors implemented
 * in their own files (nfi_hmem_cuda.cpp, nfi_hmem_rocr.cpp, nfi_hmem_neuron.cpp), which the vendor
 * table in nfi_hmem.cpp refers to. Not part of the nfi_hmem.h interface.
 */

#ifndef NFI_HMEM_VENDOR_H
#define NFI_HMEM_VENDOR_H

#include "nfi_hmem.h"

#include <cstddef>
#include <cstdint>

/* Formats a diagnostic and passes it to the sink installed with nfi_hmem_set_log_fn(), if any. */
__attribute__((format(printf, 2, 3))) void
nfi_log(enum nfi_hmem_log_level level, const char *fmt, ...);

/* Neuron (nfi_hmem_neuron.cpp): libnrt is loaded with dlopen, so it is always built. */
bool
nfi_neuron_init(void);
bool
nfi_neuron_query(const void *addr, struct nfi_hmem_info *info);

#ifdef NIXL_HAVE_CUDA_DRIVER_API
/* CUDA (nfi_hmem_cuda.cpp), through the driver API. */
bool
nfi_cuda_init(void);
bool
nfi_cuda_query(const void *addr, struct nfi_hmem_info *info);
size_t
nfi_cuda_device_bdfs(struct nfi_hmem_bus_id *out, size_t max);
int
nfi_cuda_scope_push(uint64_t device, uint64_t context, uint64_t *saved);
void
nfi_cuda_scope_pop(uint64_t saved);
#endif

#ifdef NIXL_HAVE_ROCM
/* AMD ROCr (nfi_hmem_rocr.cpp), through the linked HSA runtime. */
bool
nfi_rocr_init(void);
bool
nfi_rocr_query(const void *addr, struct nfi_hmem_info *info);
size_t
nfi_rocr_device_bdfs(struct nfi_hmem_bus_id *out, size_t max);
#endif

#endif /* NFI_HMEM_VENDOR_H */
