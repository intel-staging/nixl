# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# XPU device detection unit test — only built when Level Zero is available.

if not libze_dep.found()
    subdir_done()
endif

xpu_detect_test_bin = executable('test_xpu_detect',
    'test_xpu_detect.cpp',
    '../../../../src/api/cpp/backend/nixl_xpu_device.cpp',
    dependencies: [libze_dep, nixl_common_dep],
    include_directories: [nixl_inc_dirs, utils_inc_dirs],
    cpp_args: ['-DHAVE_ZE'],
    install: true,
)

test('test_xpu_detect', xpu_detect_test_bin)
