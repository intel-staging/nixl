/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef __ZE_H
#define __ZE_H

#include <cstddef>
#include <iostream>

/* Level Zero (Intel GPU) device memory for VRAM segments. libze_loader is dlopen'd, so a build
 * with HAVE_ZE still runs on a host without it.
 *
 * The zeAccel prefix avoids colliding with the libze_loader exports. Each function returns a
 * ze_result_t widened to int: 0 is success. */

/* Number of visible Level Zero root devices, or -1 if Level Zero is unavailable at runtime. */
int
zeAccelDeviceCount();

int
zeAccelMalloc(void **addr, size_t buffer_size, int devid = 0);
int
zeAccelFree(void *addr);

/* Fill @a count bytes at @a addr with @a val. Blocks until the fill has completed. */
int
zeAccelMemset(void *addr, int val, size_t count);

/* Copy @a len bytes of device memory at @a device_addr to @a host_addr. Blocks until the copy has
   completed. Used by --check_consistency. */
int
zeAccelMemcpyDtoH(void *host_addr, const void *device_addr, size_t len);

#define CHECK_ZE_ERROR(result, message)                                                   \
    do {                                                                                  \
        const auto _r = (result);                                                         \
        if (_r != 0) {                                                                    \
            std::cerr << "ZE: " << message << " (Error code: " << _r << ")" << std::endl; \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    } while (0)

#endif
