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

libfabric_utils_dep = [ libfabric_dep, nixl_common_deps, serdes_interface ]
libfabric_test_cpp_args = []

if cuda_dep.found()
    libfabric_utils_dep += [ cuda_dep ]
    libfabric_test_cpp_args += [ '-DCUDA_FOUND' ]
endif

if get_option('buildtype') != 'release'

    # Compute libfabric lib dir for LD_LIBRARY_PATH injection in test envs.
    # Required when libfabric_path points to a non-system install.
    lf_lib_dir = libfabric_path != '' ? libfabric_path + '/lib' : ''

    # mock function calls for testing
    libfabric_test_link_args = [
        '-Wl,--wrap=fi_getinfo',
        '-Wl,--wrap=numa_max_node',
        '-Wl,--wrap=numa_num_configured_nodes',
        '-Wl,--wrap=numa_distance',
        '-Wl,--wrap=get_mempolicy',
        '-Wl,--wrap=fi_fabric' ]

    libfabric_topology_test_bin = executable('libfabric_topology_test',
               'libfabric_topology_test.cpp',
               dependencies: libfabric_utils_dep,
               include_directories: [nixl_inc_dirs, utils_inc_dirs],
               link_with: libfabric_utils_lib,
               cpp_args: libfabric_test_cpp_args,
               link_args: libfabric_test_link_args,
               install: true)

    # install topology test files
    install_subdir('topo', strip_directory : true, install_dir : get_option('bindir'))

    rail_active_refcount_test_link_args = [
        '-Wl,--wrap=fi_getinfo',
        '-Wl,--wrap=numa_max_node',
        '-Wl,--wrap=numa_num_configured_nodes',
        '-Wl,--wrap=fi_fabric' ]

    rail_active_refcount_test_bin = executable('rail_active_refcount_test',
               'rail_active_refcount_test.cpp',
               dependencies: libfabric_utils_dep,
               include_directories: [nixl_inc_dirs, utils_inc_dirs],
               link_with: libfabric_utils_lib,
               cpp_args: libfabric_test_cpp_args,
               link_args: rail_active_refcount_test_link_args,
               install: true)

    rail_test_env = environment()
    if lf_lib_dir != ''
        rail_test_env.prepend('LD_LIBRARY_PATH', lf_lib_dir)
    endif

    test('rail_active_refcount_test', rail_active_refcount_test_bin,
         env: rail_test_env)

    # Intel XPU topology detection test — validates isIntelXpuAccel() using
    # hwloc XML files from real Intel GPU systems.
    xpu_topo_test_link_args = [
        '-Wl,--wrap=fi_getinfo',
        '-Wl,--wrap=fi_fabric' ]

    libfabric_xpu_topology_test_bin = executable('libfabric_xpu_topology_test',
               'libfabric_xpu_topology_test.cpp',
               dependencies: libfabric_utils_dep,
               include_directories: [nixl_inc_dirs, utils_inc_dirs],
               link_with: libfabric_utils_lib,
               cpp_args: libfabric_test_cpp_args,
               link_args: xpu_topo_test_link_args,
               install: true)

    xpu_topo_test_env = environment()
    if lf_lib_dir != ''
        xpu_topo_test_env.prepend('LD_LIBRARY_PATH', lf_lib_dir)
    endif

    test('libfabric_xpu_topology_test',
         libfabric_xpu_topology_test_bin,
         workdir: meson.current_source_dir() / 'topo',
         env: xpu_topo_test_env)

endif
