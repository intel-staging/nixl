# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 Intel Corporation. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Libfabric plugin unit tests

# Find gtest dependency (required for these unit tests)
gtest_test_dep = dependency('gtest', version : '>=1.11.0', required : true)

libfabric_backend_dep = declare_dependency(
    link_with: libfabric_backend_lib,
    include_directories: [nixl_inc_dirs, '../../../../src/plugins/libfabric']
)

# Collect compile flags
compile_flags = []
additional_deps = []

if cuda_dep.found()
    additional_deps += [cuda_dep]
    compile_flags += ['-DHAVE_CUDA']
endif

if synapseai_dep.found()
    additional_deps += [synapseai_dep]
    compile_flags += ['-DHAVE_SYNAPSEAI']
endif

libfabric_test_deps = [
    gtest_test_dep,
    nixl_dep,
    nixl_infra,
    nixl_common_deps,
    libfabric_utils_dep,
    libfabric_backend_dep,
    libfabric_dep,
    thread_dep,
] + additional_deps

# Test 1: Descriptor offset handling
test_descriptor_offsets = executable(
    'test_libfabric_descriptor_offsets',
    'test_libfabric_descriptor_offsets.cpp',
    dependencies: libfabric_test_deps,
    cpp_args: compile_flags,
    include_directories: [nixl_inc_dirs, utils_inc_dirs, '../../../../src/plugins/libfabric'],
    install: false,
)

test('libfabric_descriptor_offsets',
     test_descriptor_offsets,
     suite: ['unit', 'plugins', 'libfabric'],
     timeout: 60,
)

# Future tests can be added here following the same pattern:
# - test_libfabric_topology.cpp (topology discovery tests)
# - test_libfabric_rail_manager.cpp (rail management tests)
# - test_libfabric_memory_registration.cpp (memory registration tests)
# - test_libfabric_connection.cpp (connection management tests)
