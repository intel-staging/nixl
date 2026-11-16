/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Intel Corporation. All rights reserved.
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

#include "backend/backend_plugin.h"
#include "ofi_backend.h"

using ofi_plugin_t = nixlBackendPluginCreator<nixlOfiEngine>;

namespace {
nixl_b_params_t
get_ofi_backend_options() {
    nixl_b_params_t params;
    params["eq_timeout_ms"] = "100";
    return params;
}
} // namespace

#ifdef STATIC_PLUGIN_OFI
nixlBackendPlugin *
createStaticOFIPlugin() {
    return ofi_plugin_t::create(
        NIXL_PLUGIN_API_VERSION, "OFI", "0.1.0", get_ofi_backend_options(), {DRAM_SEG, VRAM_SEG});
}
#else
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return ofi_plugin_t::create(
        NIXL_PLUGIN_API_VERSION, "OFI", "0.1.0", get_ofi_backend_options(), {DRAM_SEG, VRAM_SEG});
}

extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {}
#endif
