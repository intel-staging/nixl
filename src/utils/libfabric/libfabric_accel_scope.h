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
#ifndef NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_ACCEL_SCOPE_H
#define NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_ACCEL_SCOPE_H

/*
 * nixlAccelScope and what it binds, apart from libfabric_accel.h so that headers carrying a
 * binding (the rails' post requests) need only nfi_hmem.h, not hwloc.
 */

#include "nfi_hmem.h"

#include <cstdint>

/** @brief What @ref nixlAccelScope binds: a buffer's device and its owning context. */
struct nixlAccelBinding {
    enum fi_hmem_iface iface = FI_HMEM_SYSTEM;
    uint64_t device = 0;
    uint64_t context = 0;

    nixlAccelBinding() = default;

    nixlAccelBinding(enum fi_hmem_iface i, uint64_t d, uint64_t c = 0)
        : iface(i),
          device(d),
          context(c) {}

    explicit nixlAccelBinding(const struct nfi_hmem_info &info)
        : nixlAccelBinding(info.iface, info.device, info.context) {}

    bool
    operator==(const nixlAccelBinding &other) const {
        return iface == other.iface && device == other.device && context == other.context;
    }
};

/**
 * @brief RAII wrapper over @ref nfi_hmem_scope_push_ctx and @ref nfi_hmem_scope_pop.
 *
 * Brackets operations that need the buffer's device current: registration (libfabric exports a
 * dma-buf from inside fi_mr_regattr()) and posting (a provider may copy through the host).
 * Binds the allocation's own context where the runtime reports one. Detection needs no context. A
 * no-op for runtimes without thread-current state.
 */
class nixlAccelScope {
public:
    explicit nixlAccelScope(const struct nfi_hmem_info &info)
        : nixlAccelScope(nixlAccelBinding(info)) {}

    nixlAccelScope(enum fi_hmem_iface iface, uint64_t device)
        : nixlAccelScope(nixlAccelBinding(iface, device)) {}

    explicit nixlAccelScope(const nixlAccelBinding &binding)
        : scope_{},
          binding_(binding),
          ok_(nfi_hmem_scope_push_ctx(binding.iface, binding.device, binding.context, &scope_) ==
              0) {}

    ~nixlAccelScope() {
        nfi_hmem_scope_pop(&scope_);
    }

    nixlAccelScope(const nixlAccelScope &) = delete;
    nixlAccelScope &
    operator=(const nixlAccelScope &) = delete;

    /** @brief False when the runtime refused the binding. */
    bool
    ok() const {
        return ok_;
    }

    /** @brief Whether this scope already binds what @p binding needs. */
    bool
    holds(const nixlAccelBinding &binding) const {
        return ok_ && binding_ == binding;
    }

    /** @brief Whether this scope already binds the device and context @p info needs. */
    bool
    holds(const struct nfi_hmem_info &info) const {
        return holds(nixlAccelBinding(info));
    }

private:
    struct nfi_hmem_scope scope_;
    nixlAccelBinding binding_;
    bool ok_;
};

#endif // NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_ACCEL_SCOPE_H
