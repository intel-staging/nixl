# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Amazon.com, Inc. and affiliates.
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

# Source files.
#
# nfi_hmem.cpp is the libfabric HMEM shim: everything libfabric knows about attributing a device
# pointer but does not export. Keep it free of nixl types, nixl logging and hwloc -- see the header
# for the full set of constraints, which exist so it stays donatable upstream.
libfabric_utils_sources = files(
    'libfabric_rail.cpp',
    'libfabric_rail_manager.cpp',
    'libfabric_common.cpp',
    'libfabric_topology.cpp',
    'libfabric_accel.cpp',
    'nfi_hmem.cpp',
    # More implementation files will be added as we create them
)

# Header files
libfabric_utils_headers = files(
    'libfabric_rail.h',
    'libfabric_rail_manager.h',
    'libfabric_common.h',
    'libfabric_topology.h',
    'libfabric_accel.h',
    'nfi_hmem.h',
)

# Find hwloc dependency
hwloc_dep = dependency('hwloc', required: true)

# Find numa dependency
numa_dep = dependency('numa', required: true)

# dlopen/dlsym, used by nfi_hmem.cpp to load libnrt and libze_loader at runtime
dl_dep = dependency('dl', required: true)

# pthread, for the shim's one-time vendor initialisation lock
thread_dep_libfabric = dependency('threads', required: true)

# Set up dependencies and compile args
libfabric_utils_deps = [
    libfabric_dep,
    hwloc_dep,
    numa_dep,
    dl_dep,
    thread_dep_libfabric,
    absl_log_dep,
]

libfabric_utils_cpp_args = []

# Detect optional libfabric features at compile time to enable/disable
# functionality based on the installed libfabric version
if libfabric_dep.found()
    # Check for FI_OPT_EFA_USE_UNSOLICITED_WRITE_RECV (libfabric >= 2.3)
    if cpp.has_header_symbol('rdma/fi_ext.h',
                             'FI_OPT_EFA_USE_UNSOLICITED_WRITE_RECV',
                             dependencies: libfabric_dep)
        libfabric_utils_cpp_args += ['-DHAVE_FI_OPT_EFA_USE_UNSOLICITED_WRITE_RECV']
        message('libfabric supports FI_OPT_EFA_USE_UNSOLICITED_WRITE_RECV')
    else
        message('libfabric does not support FI_OPT_EFA_USE_UNSOLICITED_WRITE_RECV')
    endif
endif

# Add CUDA support if available
if cuda_dep.found()
    libfabric_utils_deps += [cuda_dep]
    libfabric_utils_cpp_args += ['-DHAVE_CUDA']
endif

# Add Level Zero (Intel Xe / XPU) support if the headers are available. Headers only: the
# loader is dlopen'd, so there is no dependency to add here.
if have_ze
    libfabric_utils_cpp_args += ['-DHAVE_ZE']
endif

# Create static library
libfabric_utils_lib = static_library(
    'nixl_libfabric_utils',
    libfabric_utils_sources,
    dependencies: libfabric_utils_deps,
    cpp_args: libfabric_utils_cpp_args,
    include_directories: [
        nixl_inc_dirs,
        utils_inc_dirs,
    ],
    install: false,
)

# Declare dependency for other parts of the build
libfabric_utils_dep = declare_dependency(
    link_with: libfabric_utils_lib,
    include_directories: include_directories('.'),
)
