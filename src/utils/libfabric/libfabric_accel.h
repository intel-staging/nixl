/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 Amazon.com, Inc. and affiliates.
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

#include "nixl_types.h"

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>

#include <hwloc.h>
#include <cstdint>
#include <string>
#include <vector>

/**
 * @brief nixl's side of the accelerator boundary.
 *
 * Pointer attribution, device enumeration and fi_mr_attr population come from libfabric's hmem
 * query extension (fi_open("hmem"), rdma/fi_ext.h), so this layer and the provider cannot reach
 * different conclusions about one pointer. What libfabric does not provide stays here:
 *
 *   1. **Neuron attribution.** libfabric has no Neuron pointer probe, so a Neuron pointer is
 *      identified through libnrt, which reports the attached EFA device rather than the
 *      accelerator -- see @ref nixlAccelBusKind.
 *   2. **CUDA context binding.** libfabric does no context management on the data path, so the
 *      thread-current context its CUDA calls need is established by @ref nixlAccelScope.
 *   3. **hwloc matching and rail-selection policy**, which are nixl decisions.
 */

/****************************************
 * Pointer attribution
 *****************************************/

/**
 * @brief Whose PCI address @ref nixlAccelInfo::bus holds.
 *
 * A CUDA or Level Zero pointer reports the GPU's own address; a Neuron pointer reports the address
 * of the NIC attached to that accelerator, because that is what libnrt offers -- a Trainium card
 * carries its EFA device onboard.
 */
enum class nixlAccelBusKind { NONE, ACCELERATOR, NIC };

/** @brief What is known about one registered buffer. */
struct nixlAccelInfo {
    /** FI_HMEM_SYSTEM when no runtime claimed the pointer. */
    enum fi_hmem_iface iface = FI_HMEM_SYSTEM;
    /** Encoded for the fi_mr_attr::device arm @ref iface selects; not a nixl device index. */
    uint64_t device = 0;
    nixlAccelBusKind bus_kind = nixlAccelBusKind::NONE;
    /** Valid only when @ref bus_kind is not NONE. */
    struct fi_pci_attr bus = {};
};

/**
 * @brief Identifies which accelerator runtime owns @p addr.
 *
 * Host memory is a successful FI_HMEM_SYSTEM answer, not an error. Talks to the vendor runtime, so
 * call it once per registration and keep the result.
 */
nixl_status_t
accelQueryPtr(const void *addr, nixlAccelInfo &info);

/** @brief Sets fi_mr_attr::iface and the device union arm @p info selects; nothing else. */
void
accelFillMrAttr(const nixlAccelInfo &info, struct fi_mr_attr &attr);

/****************************************
 * Runtime enablement
 *****************************************/

/**
 * @brief The interfaces nixl handles, in libfabric's probe order (descending fi_hmem_iface).
 *
 * Says nothing about whether a runtime is usable; ask @ref accelIfaceAvailable for that.
 */
const std::vector<enum fi_hmem_iface> &
nixlAccelKnownIfaces();

/** @brief Whether @p iface is usable for registration in this process. */
bool
accelIfaceAvailable(enum fi_hmem_iface iface);

/** @brief Whether any interface in @ref nixlAccelKnownIfaces is usable. */
bool
accelAvailable();

/**
 * @brief First usable interface in probe order, or FI_HMEM_SYSTEM.
 *
 * For logging, and as the fallback for a VRAM pointer no runtime claimed. Not a substitute for
 * per-pointer attribution: on a host with two runtimes it names only one.
 */
enum fi_hmem_iface
accelPrimaryIface();

/** @brief Short name of @p iface ("CUDA", "ZE", "SYSTEM", ...), from libfabric's fi_tostr. */
const char *
accelIfaceName(enum fi_hmem_iface iface);

/****************************************
 * PCI addresses
 *****************************************/

/**
 * @brief Renders a PCI address the way hwloc does, so both sides key alike.
 *
 * hwloc writes "%x:%02x:%02x.%x" -- an unpadded domain, everything else padded.
 */
std::string
accelBusIdString(const struct fi_pci_attr &bus);

/** @brief @ref accelBusIdString for @p info, or an empty string when it has no address. */
std::string
accelBusIdString(const nixlAccelInfo &info);

/****************************************
 * hwloc matching
 *****************************************/

/**
 * @brief The interface whose vendor owns @p obj, or FI_HMEM_SYSTEM if none does.
 *
 * NVIDIA and Neuron are matched from hwloc alone, which keeps discovery independent of
 * CUDA_VISIBLE_DEVICES. Intel is matched against the Level Zero devices libfabric enumerated,
 * because a discrete Intel GPU and Intel integrated graphics share vendor ID and PCI class.
 */
enum fi_hmem_iface
accelIfaceOfHwlocObj(hwloc_obj_t obj);

/****************************************
 * Rail-selection policy
 *****************************************/

/**
 * @brief Whether @p iface's devices take part in PCIe-proximity NIC grouping.
 *
 * Neuron devices are accelerators for counting purposes but stay out of grouping: a Neuron pointer
 * reports a NIC address and resolves through the NIC map.
 */
bool
accelIfaceIsGroupable(enum fi_hmem_iface iface);

/**
 * @brief Whether a VRAM_SEG pointer no runtime claimed is a caller error.
 *
 * True when CUDA is usable: device memory CUDA does not recognise is a bug in the caller. Neuron
 * and Level Zero fall back to registering on all rails.
 */
bool
accelRequiresStrictAttribution();

/****************************************
 * Context scope
 *****************************************/

/**
 * @brief Makes the buffer's CUDA device current on this thread for the scope's lifetime.
 *
 * Registration needs a context because libfabric exports a dmabuf handle from inside
 * fi_mr_regattr() for providers configured that way; posting is bracketed too, because a provider
 * may take a host-copy path for small transfers. Attribution needs none.
 *
 * Pushes the device's primary context and pops it on destruction, so an application thread's own
 * binding survives. A no-op for host memory and for runtimes with no thread-current state, so
 * callers can construct one unconditionally.
 */
class nixlAccelScope {
public:
    explicit nixlAccelScope(const nixlAccelInfo &info) : nixlAccelScope(info.iface, info.device) {}

    nixlAccelScope(enum fi_hmem_iface iface, uint64_t device);
    ~nixlAccelScope();

    nixlAccelScope(const nixlAccelScope &) = delete;
    nixlAccelScope &
    operator=(const nixlAccelScope &) = delete;

    /** @brief False only when the runtime actively refused; nothing to bind is success. */
    bool
    ok() const {
        return ok_;
    }

private:
    bool pushed_ = false;
    int device_ = 0;
    bool ok_ = true;
};

#endif // NIXL_SRC_UTILS_LIBFABRIC_LIBFABRIC_ACCEL_H
