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
 * nfi_hmem vendor: AWS Neuron, through a dlopen'd libnrt.
 */

#include "nfi_hmem_vendor.h"

#include <dlfcn.h>

#include <cstdio>

/****************************************
 * Neuron
 *****************************************/

static int (*nfi_nrt_get_attached_efa_bdf)(const void *va, char *efa_bdf, size_t *len);

bool
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

bool
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
