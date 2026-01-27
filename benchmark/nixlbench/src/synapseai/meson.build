# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

synapseaiutils_sources = [
  'synapse_utils.cpp',
  'synapse_utils.h',
]

synapseaiutils_deps = [
  synapseai_dep
]

synapseaiutils_lib = static_library('synapseaiutils',
  synapseaiutils_sources,
  dependencies: synapseaiutils_deps,
  include_directories: inc_dir
)
synapseaiutils_dep = declare_dependency(
  link_with: synapseaiutils_lib,
  dependencies: synapseaiutils_deps,
  include_directories: inc_dir
)
