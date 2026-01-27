/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <iostream>
#include <mutex>
#include <synapse_api.h>

namespace Synapseaiutils {
int
init_synapse_device();
synDeviceId
get_device_handle();
uint64_t
allocate_synapse_memory(size_t len, void *host_buffer);
void
free_synapse_memory(uint64_t ptr);
void
deinit_synapse_device();
void
copy_from_device_buffer(uint64_t device_buffer, void *host_buffer, size_t len);
} // namespace Synapseaiutils
