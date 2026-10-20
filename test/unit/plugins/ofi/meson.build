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

gtest_dep = dependency('gtest', version : '>=1.11.0', required : true)

ofi_backend_dep = declare_dependency(link_with: ofi_backend_lib, include_directories: [nixl_inc_dirs, '../../../../src/plugins/ofi'])

ofi_backend_test = executable('ofi_backend_test',
        'ofi_backend_test.cpp',
        dependencies: [nixl_dep, nixl_infra, nixl_common_dep, ofi_backend_dep, libfabric_dep, thread_dep, gtest_dep],
        include_directories: [nixl_inc_dirs, utils_inc_dirs, '../../../../src/plugins/ofi'],
        install: true)

test('ofi_backend_test', ofi_backend_test)

