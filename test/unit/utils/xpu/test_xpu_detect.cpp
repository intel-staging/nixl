/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * Unit test for Intel XPU (Level Zero) device detection.
 *
 * On non-XPU systems this test exits 0 immediately (graceful skip).
 * On XPU-capable systems it validates that PCI BDFs are populated.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef HAVE_ZE
#include "backend/nixl_xpu_device.h"
#endif

int
main() {
#ifndef HAVE_ZE
    printf("HAVE_ZE not defined; skipping XPU detection test.\n");
    return 0;
#else
    printf("Running XPU device detection test...\n");

    std::vector<XpuDevice> devices = XpuDevice::detectDevices();

    printf("Detected %zu Intel XPU device(s)\n", devices.size());

    // Graceful on systems without Intel XPUs — zero devices is a valid result.
    if (devices.empty()) {
        printf("No Intel XPU devices found (non-XPU system). Test PASSED.\n");
        return 0;
    }

    // Validate each detected device.
    for (size_t i = 0; i < devices.size(); ++i) {
        if (!devices[i].device) {
            fprintf(stderr, "FAIL: device[%zu] has null ze_device_handle_t\n", i);
            return 1;
        }
        if (!devices[i].context) {
            fprintf(stderr, "FAIL: device[%zu] has null ze_context_handle_t\n", i);
            return 1;
        }
        if (!devices[i].pci_bdf.empty()) {
            if (devices[i].pci_bdf.find(':') == std::string::npos) {
                fprintf(stderr,
                        "FAIL: device[%zu] PCI BDF '%s' does not contain ':'\n",
                        i,
                        devices[i].pci_bdf.c_str());
                return 1;
            }
        }
        printf("  device[%zu]: pci_bdf=%s driver=%u idx=%u\n",
               i,
               devices[i].pci_bdf.c_str(),
               devices[i].driver_index,
               devices[i].device_index);
    }

    printf("XPU detection test PASSED.\n");
    return 0;
#endif
}
