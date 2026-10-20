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

if not libfabric_dep.found()
    message('Libfabric not found, skipping OFI plugin')
    subdir_done()
endif

if 'OFI' in static_plugins
    ofi_backend_lib = static_library('OFI',
               'ofi_backend.cpp', 'ofi_backend.h', 'ofi_plugin.cpp',
               dependencies: [nixl_infra, nixl_common_dep, libfabric_dep, dependency('dl', required: true)],
               include_directories: [nixl_inc_dirs, utils_inc_dirs],
               install: false,
               name_prefix: 'libplugin_')
else
    ofi_backend_lib = shared_library('OFI',
               'ofi_backend.cpp', 'ofi_backend.h', 'ofi_plugin.cpp',
               dependencies: [nixl_infra, nixl_common_dep, libfabric_dep, dependency('dl', required: true)],
               include_directories: [nixl_inc_dirs, utils_inc_dirs],
               install: true,
               cpp_args : ['-fPIC'],
               name_prefix: 'libplugin_',
               install_dir: plugin_install_dir)

    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
                    'echo "OFI=' + ofi_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
                    check: true
                )
    endif
endif

ofi_backend_interface = declare_dependency(link_with: ofi_backend_lib)
