/*
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

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_UTILS_ZE_H
#define NIXL_BENCHMARK_NIXLBENCH_SRC_UTILS_ZE_H

#include <cstddef>
#include <iostream>

/* Intel GPU device memory through Level Zero. All functions return a
   ze_result_t value (0 on success). */

int
zeMalloc(void **addr, size_t buffer_size, int devid);
int
zeFree(void *addr);

int
zeMemset(void *addr, int val, size_t count);
int
zeMemcpyDeviceToHost(void *dest, const void *src, size_t count);

#define CHECK_ZE_ERROR(result, message)                                                           \
    do {                                                                                          \
        const auto _r = (result);                                                                 \
        if (_r != 0) {                                                                            \
            std::cerr << "ZE: " << message << " (Error code: " << std::hex << std::showbase << _r \
                      << std::dec << std::noshowbase << ")" << std::endl;                         \
            exit(EXIT_FAILURE);                                                                   \
        }                                                                                         \
    } while (0)

#endif
