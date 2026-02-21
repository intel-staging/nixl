<!---
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-FileCopyrightText: Copyright (c) 2025-2026 Intel Corporation. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Intel XPU Support in NIXL

NIXL supports Intel XPU devices (Intel Data Center GPU Max and Flex series) through the
Level Zero API (`libze_loader`) and libfabric's `FI_HMEM_ZE` memory registration interface.

## Prerequisites

- Intel GPU Arc series (or any Level Zero-capable device)
- [Level Zero runtime and loader](https://github.com/oneapi-src/level-zero):
  `libze_loader.so` must be installed (typically via the Intel GPU driver package)
- libfabric built with ZE HMEM support (`--with-ze` / `FI_HMEM_ZE` capability available)
- A libfabric provider that supports `FI_HMEM` capabilities (e.g. `verbs;ofi_rxm` or EFA)

## Build

### 1. Install Level Zero runtime and headers

**Ubuntu / Debian** (via Intel GPU driver PPA):
```bash
# Add the Intel GPU driver repository if not already present
# See https://dgpu-docs.intel.com/ for the current repo setup instructions
sudo apt-get install libze1 libze-dev libze-intel-gpu1
```

**RHEL / Rocky / SLES** (via Intel oneAPI repository):
```bash
sudo dnf install level-zero level-zero-devel
```

Verify the loader is findable by pkg-config:
```bash
pkg-config --modversion libze_loader
```

### 2. Install libfabric with ZE HMEM support

The system libfabric must be built with `--with-ze`. Verify:
```bash
fi_info --version    # should list ze in HMEM interfaces
```

### 3. Build NIXL

The build system auto-detects `libze_loader`
(`-Denable_xpu_backend=auto`, the default):

```bash
meson setup build
ninja -C build
```

To force-enable (fail the configure step if Level Zero is missing):
```bash
meson setup build -Denable_xpu_backend=true
```

To disable explicitly:
```bash
meson setup build -Denable_xpu_backend=false
```

When `libze_loader` is found, the build defines `-DHAVE_ZE` and links
`libze_loader` into the LIBFABRIC plugin. The configure summary will
show:

```
Accelerator Support
  Intel XPU (Level Zero): YES
```

## C++ API usage

```cpp
#include "nixl.h"

// 1. Allocate device memory with Level Zero
ze_device_mem_alloc_desc_t alloc_desc = { ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC };
void *buf = nullptr;
zeMemAllocDevice(ze_context, &alloc_desc, size, 64, ze_device, &buf);

// 2. Register with NIXL using VRAM_SEG (Intel XPU uses the same type as CUDA/Neuron).
//    devId is the device ordinal (0-based); the backend resolves
//    the ze_device_handle_t internally via zeMemGetAllocProperties.
nixl_reg_dlist_t reg_list(VRAM_SEG);
reg_list.addDesc({(uintptr_t)buf, size, /*devId=*/0, ""});
agent->registerMem(reg_list);

// 3. Transfer / deregister / free as normal
// ...
agent->deregisterMem(reg_list);
zeMemFree(ze_context, buf);
```

## Python API usage

```python
import nixl

agent = nixl.nixlAgent("my_agent", nixl.nixlAgentConfig())
agent.create_backend("LIBFABRIC")

# Allocate Intel XPU memory (e.g. via PyZE or ctypes + ze_api)
# buf_ptr, buf_len, device_id = ...

# Pass "XPU" or "VRAM" — both resolve to VRAM_SEG internally.
descs = nixl.nixlRegDList(nixl.nixl_mem_t.VRAM_SEG)
descs.addDesc((buf_ptr, buf_len, device_id, b""))
agent.register_memory(descs)
```

## PCIe topology and rail selection

The NIXL libfabric backend performs NUMA-aware NIC rail selection for XPU memory the same
way it does for NVIDIA GPU memory:

1. At registration time, `XpuDevice::getDeviceForPtr()` uses `zeMemGetAllocProperties()`
   to identify which `ze_device_handle_t` owns the buffer.
2. The matching `XpuDevice` entry provides the PCIe BDF (`domain:bus:dev.func`).
3. `nixlLibfabricTopology::getEfaDevicesForPci()` maps the BDF to the topologically
   closest set of NIC rails using hwloc PCIe proximity data.
4. Memory is registered only on those rails, minimising PCIe congestion.

## Known limitations

- Each Level Zero driver creates one shared context; sub-device contexts are not yet supported.
- Data consistency checks in nixlbench (`--check_consistency`) are not implemented for XPU
  memory (the device pointer is not host-accessible without a staging copy).
- Only the LIBFABRIC backend supports Intel XPU (VRAM_SEG with FI_HMEM_ZE); UCX XPU
  support is not yet implemented.
- `FI_HMEM_DEVICE_ONLY` flag is not set by default; add it if your libfabric provider requires
  it for device-only allocations.
