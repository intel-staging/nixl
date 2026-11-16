# NIXL OFI Plugin

Libfabric backend for Intel Gaudi. It registers Gaudi HBM (`VRAM_SEG`) through the
SynapseAI dmabuf export and moves it with libfabric RMA, and registers host memory
(`DRAM_SEG`) directly. vllm-gaudi selects it with `VLLM_NIXL_BACKEND=OFI`.

## Requirements

- libfabric built with SynapseAI HMEM support (`--with-synapseai`); `install_nixl.py`
  builds one and bundles it in the wheel.
- SynapseAI headers at build time (`habanalabs/synapse_api.h`); without them the plugin is
  skipped. `libSynapse.so` and `libhl-thunk.so` at run time; without them only `DRAM_SEG`
  is advertised.

## Runtime configuration

- `FI_PROVIDER` selects the provider: `verbs` (default, `verbs;ofi_rxm`), `tcp` or `shm`.
- Backend parameter `eq_timeout_ms` (default 100, range 0-60000).
- In containers, pass `--device=/dev/infiniband/rdma_cm` as well as `/dev/infiniband`;
  without it `verbs;ofi_rxm` fails in `rdma_create_id`.

## Known limitations

- **Notifications are not implemented.** `genNotif` and `getNotifs` are no-ops and
  notifications attached to transfers are dropped, although `supportsNotif()` returns
  true; the backend logs a warning once. With vLLM's NIXL connector, prefill frees KV
  blocks only when their lease expires instead of when decode has read them. Transfers
  that depend on notifications, such as push mode, where the receiver waits for a
  notification that the data has arrived, do not work.
- `VRAM_SEG` is Gaudi only; CUDA and ZE device memory are not supported. Registering
  `VRAM_SEG` without SynapseAI fails with `NIXL_ERR_NOT_SUPPORTED`.
- Only `verbs` can move HBM. `tcp` and `shm` move `DRAM_SEG` only.
- nixlbench has no OFI backend on this branch.
