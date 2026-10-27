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

/**
 * @file nfi_hmem.h
 * @brief Heterogeneous-memory helpers that libfabric uses internally but does not export.
 *
 * Identifies which accelerator runtime owns a pointer and fills fi_mr_attr::iface and
 * fi_mr_attr::device for fi_mr_regattr(). The interface is C-compatible and depends only on
 * libfabric's public headers; vendor runtimes are loaded with dlopen. All functions are
 * thread-safe, and each vendor is initialised once, on first use.
 */

#ifndef NFI_HMEM_H
#define NFI_HMEM_H

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief A PCI address, unformatted. */
struct nfi_hmem_bus_id {
    uint16_t domain;
    uint8_t bus;
    uint8_t slot;
    uint8_t func;
};

/** @brief Which device @ref nfi_hmem_info::bus describes. */
enum nfi_hmem_bus_kind {
    /** No PCI address available. */
    NFI_HMEM_BUS_NONE = 0,
    /** The accelerator's own PCI address. */
    NFI_HMEM_BUS_ACCELERATOR,
    /** The PCI address of a NIC attached to the accelerator. */
    NFI_HMEM_BUS_NIC
};

/** @brief What the vendor runtimes know about one pointer. */
struct nfi_hmem_info {
    /** Owning interface; FI_HMEM_SYSTEM when no vendor claimed the pointer. */
    enum fi_hmem_iface iface;

    /**
     * Device ordinal, already encoded for the fi_mr_attr::device arm @p iface selects (e.g.
     * fi_hmem_ze_device(0, index) for Level Zero). Not a nixl device index.
     */
    uint64_t device;

    /** Which device @ref bus describes. */
    enum nfi_hmem_bus_kind bus_kind;

    /** PCI address, valid only when @ref bus_kind is not NFI_HMEM_BUS_NONE. */
    struct nfi_hmem_bus_id bus;

    /**
     * Size of the dma-buf exported for this pointer when it is larger than the allocation, i.e. the
     * allocation is a slice of a pooled buffer object that libfabric would register at the wrong
     * offset. Level Zero only; 0 otherwise.
     */
    uint64_t pooled_dmabuf_size;

    /** Size of the allocation; set together with @ref pooled_dmabuf_size. */
    uint64_t alloc_size;
};

/** @brief Diagnostic severity. */
enum nfi_hmem_log_level {
    NFI_HMEM_LOG_ERROR = 0,
    NFI_HMEM_LOG_WARN,
    NFI_HMEM_LOG_INFO,
    NFI_HMEM_LOG_DEBUG
};

/** @brief Diagnostic sink. @p msg is a single NUL-terminated line. May be called from any thread.
 */
typedef void (*nfi_hmem_log_fn)(enum nfi_hmem_log_level level, const char *msg);

/**
 * @brief Installs the diagnostic sink. NULL (the default) silences output.
 *
 * Install it before the first query to see vendor initialisation messages.
 */
void
nfi_hmem_set_log_fn(nfi_hmem_log_fn fn);

/**
 * @brief Identifies which accelerator runtime owns @p addr.
 *
 * Vendors are probed in the same order as libfabric's ofi_get_hmem_iface(), so this and a provider
 * that derives the iface itself (cxi, rxm, opx) agree about a pointer. Host memory is reported as
 * FI_HMEM_SYSTEM with a success return. Talks to the vendor runtime: call once per registration.
 *
 * @param addr Pointer to identify.
 * @param[out] info Result. Fully overwritten.
 * @return 0 on success, -EINVAL if @p addr or @p info is NULL.
 */
int
nfi_hmem_query_ptr(const void *addr, struct nfi_hmem_info *info);

/**
 * @brief Sets fi_mr_attr::iface and the fi_mr_attr::device arm that @p info selects.
 *
 * Only iface and device are written. For FI_HMEM_SYSTEM device is left unset.
 */
void
nfi_hmem_fill_mr_attr(const struct nfi_hmem_info *info, struct fi_mr_attr *attr);

/**
 * @brief Whether any accelerator runtime is usable in this process.
 *
 * Independent of the libfabric provider; check provider FI_HMEM support separately.
 */
bool
nfi_hmem_available(void);

/** @brief Whether the runtime for @p iface is usable. Initialises that vendor on first call. */
bool
nfi_hmem_iface_available(enum fi_hmem_iface iface);

/** @brief First usable runtime in probe order, or FI_HMEM_SYSTEM if there is none. */
enum fi_hmem_iface
nfi_hmem_primary_iface(void);

/**
 * @brief Every interface with a vendor entry, in probe order, usable or not.
 *
 * @param[out] count Number of entries. Must be non-NULL.
 * @return Static array of @p count entries.
 */
const enum fi_hmem_iface *
nfi_hmem_ifaces(size_t *count);

/**
 * @brief Short name of @p iface ("ZE", "CUDA", "SYSTEM", ...), from fi_tostr() without the
 *        "FI_HMEM_" prefix.
 *
 * @return Static string, never NULL; "Unknown" for a value libfabric does not recognise.
 */
const char *
nfi_hmem_iface_name(enum fi_hmem_iface iface);

/**
 * @brief PCI addresses of the RDMA-capable accelerators the runtime for @p iface enumerated.
 *
 * For Level Zero these are the discrete GPUs: an Intel discrete GPU and an integrated one share PCI
 * vendor and class, so only the driver can tell them apart. Returns 0 for a vendor that cannot
 * enumerate or is unavailable. Initialises the vendor on first call.
 *
 * @param[out] out Destination, or NULL to query the count only.
 * @param max Capacity of @p out in entries.
 * @return Total number of devices, which may exceed @p max; at most @p max are written.
 */
size_t
nfi_hmem_device_bdfs(enum fi_hmem_iface iface, struct nfi_hmem_bus_id *out, size_t max);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NFI_HMEM_H */
