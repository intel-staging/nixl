# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright 2025 Intel Corporation
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

utils_inc = include_directories('.')

synapseai_utils_lib = static_library(
  'synapseai_utils',
  ['synapseai_utils.cpp'],
  include_directories: utils_inc,
  dependencies: [ synapseai_dep ],
)

synapseai_utils_dep = declare_dependency(
  link_with: synapseai_utils_lib,
  include_directories: utils_inc,
  dependencies: [ synapseai_dep ],
)

# Export to parent (test/) scope
test_synapseai_utils_dep = synapseai_utils_dep

