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

namespace
{
    static const char *PLUGIN_NAME = "OFI";
    static const char *PLUGIN_VERSION = "0.1.0";
    
    // create a new OFI backend engine instance
    static nixlBackendEngine *
    create_ofi_engine (const nixlBackendInitParams *init_params) {
        return new nixlOFI_Engine (init_params);
    }
    
    static void
    destroy_ofi_engine (nixlBackendEngine *engine) {
        delete engine;
    }

    static const char *
    get_plugin_name() {
        return PLUGIN_NAME;
    }
    
    static const char *
    get_plugin_version() {
        return PLUGIN_VERSION;
    }
    
    static nixl_b_params_t
    get_backend_options() {
        nixl_b_params_t params;
        params["provider"] = "verbs";  // Default to verbs as per requirements
        params["eq_timeout_ms"] = "100";
        return params;
    }
    
    static nixl_mem_list_t
    get_backend_mems() {
        nixl_mem_list_t mems;
        mems.push_back (DRAM_SEG);
        mems.push_back (VRAM_SEG);
        return mems;
    }

    // Static plugin structure
    static nixlBackendPlugin plugin = {NIXL_PLUGIN_API_VERSION,
                                       create_ofi_engine,
                                       destroy_ofi_engine,
                                       get_plugin_name,
                                       get_plugin_version,
                                       get_backend_options,
                                       get_backend_mems};
} // namespace

#ifdef STATIC_PLUGIN_OFI

nixlBackendPlugin *
createStaticOFIPlugin() {
    return &plugin; // Return the static plugin instance
}

#else

// Plugin initialization function
extern "C" NIXL_PLUGIN_EXPORT nixlBackendPlugin *
nixl_plugin_init() {
    return &plugin;
}

// Plugin cleanup function
extern "C" NIXL_PLUGIN_EXPORT void
nixl_plugin_fini() {
    // Cleanup
}

#endif
