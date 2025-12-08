#!/usr/bin/env python3
"""
NIXL API Performance Benchmark - Receiver

Connects to sender over ZMQ, retrieves metadata, allocates local KV cache,
performs data transfers via NIXL, validates integrity, and prints benchmark
results. Designed to be run standalone on the receiver node.
"""

import argparse
import logging
import os
import time
import uuid
import hashlib
from typing import List, Iterator

import msgspec
import torch
import zmq
import pandas as pd

# Open Device for UCX backend
try:
    import habana_frameworks.torch as htorch  # noqa: F401  (kept for compatibility)
    import habana_frameworks.torch.utils.experimental as htexp

    test_tensor = torch.tensor([1.0], device="hpu")
    del test_tensor
except:
    pass

# ==============================================================================
# Configuration
# ==============================================================================

ZMQ_BASE_PORT = 15555
GET_META_MSG = b"get_meta_msg"
SHUTDOWN_MSG = b"shutdown_msg"

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s][%(processName)s][%(asctime)s] %(message)s",
)

# Assuming 'nixl_agent' is the class name provided by the library.
from nixl._api import nixl_agent as NixlAgent
from nixl._api import nixl_agent_config


class NixlAgentMetadata(msgspec.Struct, omit_defaults=True, dict=True):
    """Metadata structure exchanged between sender and receiver."""

    engine_id: str
    agent_metadata: bytes
    kv_caches_base_addr: list[int]
    num_blocks: int
    block_len: int
    block_hashes: list[str]


# ==============================================================================
# Helpers
# ==============================================================================


def get_data_ptr(t: torch.Tensor, device_type: str = "cpu") -> int:
    if device_type == "hpu":
        addr = htexp._data_ptr(t)
    else:
        addr = t.data_ptr()
    logging.info(f"Data ptr of tensor on device {t.device}: 0x{addr:x}")
    return addr


def compute_block_hash(kv_cache: torch.Tensor, block_len: int, block_idx: int) -> str:
    elem_per_blk = block_len // kv_cache.element_size()
    start = block_idx * elem_per_blk
    end = start + elem_per_blk
    block_bytes = kv_cache[start:end].cpu().numpy().tobytes()
    return hashlib.sha256(block_bytes).hexdigest()


def allocate_kv_cache_receiver(
    num_blocks: int, block_len: int, dtype: torch.dtype, device: str
) -> torch.Tensor:
    """Allocate a KV cache buffer on the given device for receiver."""
    total_bytes = num_blocks * block_len
    num_elements = total_bytes // dtype.itemsize
    logging.info(
        f"Allocating KV cache (receiver): {total_bytes / 1e6:.2f} MB "
        f"({num_elements:,} elements, {dtype}, {device})"
    )
    return torch.zeros(num_elements, dtype=dtype, device=device)


def create_xfer_descs(
    agent: NixlAgent, base_addr: int, num_blocks: int, block_len: int, mem_type: str
):
    """Create transfer descriptors for a block range."""
    blocks_data = [(base_addr + i * block_len, block_len, 0) for i in range(num_blocks)]
    return agent.get_xfer_descs(blocks_data, mem_type)


def get_block_desc_ids(num_total_blocks: int, block_ids: Iterator[int]) -> List[int]:
    """Maps logical block IDs to the indices of their transfer descriptors."""
    result = list(block_ids)
    return result


def read_blocks(
    block_ids: Iterator[int],
    agent: NixlAgent,
    local_xfer_handle: str,
    remote_xfer_handle: str,
    sender_meta: NixlAgentMetadata,
):
    """Read blocks from the sender's KV cache using NIXL."""
    block_ids_list = list(block_ids)
    if not block_ids_list:
        logging.warning("No block IDs provided for transfer.")
        return 0.0, 0

    local_ids = get_block_desc_ids(sender_meta.num_blocks, iter(block_ids_list))
    remote_ids = get_block_desc_ids(sender_meta.num_blocks, iter(block_ids_list))

    t0 = time.perf_counter_ns()
    xfer_handle = agent.make_prepped_xfer(
        "READ",
        local_xfer_handle,
        local_ids,
        remote_xfer_handle,
        remote_ids,
    )
    agent.transfer(xfer_handle)

    while agent.check_xfer_state(xfer_handle) != "DONE":
        time.sleep(0.00001)

    agent.release_xfer_handle(xfer_handle)
    del xfer_handle
    t1 = time.perf_counter_ns()

    return (t1 - t0) / 1e6, len(local_ids) * sender_meta.block_len  # ms, bytes


def summary(
    latencies: List[float],
    sender_meta: NixlAgentMetadata,
    args: argparse.Namespace,
    total_data_transferred: int,
    sustained_throughput_gbps: float,
    total_elapsed_seconds: float,
    agent: NixlAgent,
):
    total_latency_time = sum(latencies)
    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0

    print("\n" + "=" * 60)
    print(" NIXL API Performance Benchmark Results")
    print("=" * 60)
    print(f" Hardware Memory Type:    {args.nixl_memory_type}")
    print(f" Device Type:             {args.device_type}")
    backend_names = (
        list(agent.backends.keys())
        if isinstance(agent.backends, dict)
        else [str(agent.backends)]
    )
    print(f" NIXL Backend:            {', '.join(backend_names)}")
    print(f" Total Iterations:        {args.num_iterations:,}")
    print(f" Blocks per Transfer:     {args.blocks_per_xfer:,}")
    print(
        f" Data per Transfer:       {(args.blocks_per_xfer * sender_meta.block_len) / 1e6:.2f} MB"
    )
    print("-" * 60)
    print(f" Average Latency:         {avg_latency_ms:.3f} ms")
    print(f" Sustained Throughput:    {sustained_throughput_gbps:.3f} GB/s")
    print(f" Total Data Transferred:  {total_data_transferred / 1e6:.2f} MB")
    print(f" Total Elapsed Time:      {total_elapsed_seconds:.3f} s")
    print(f" Sum of Transfer Times:   {total_latency_time:.1f} ms")
    print(
        f" Overhead Time:           {(total_elapsed_seconds * 1000 - total_latency_time):.1f} ms"
    )
    print("-" * 60)
    print(" Latency Distribution (ms):")
    stats = pd.Series(latencies).describe()
    print(f"   Count:      {stats['count']:8.0f}")
    print(f"   Mean:       {stats['mean']:8.3f} ms")
    print(f"   Max:        {stats['max']:8.3f} ms")
    print("-" * 60)
    print(" Transfer Efficiency:")
    efficiency_pct = (total_latency_time / (total_elapsed_seconds * 1000)) * 100
    print(f"   Active Transfer Time:    {efficiency_pct:.1f}%")
    print(f"   Overhead Time:           {100 - efficiency_pct:.1f}%")
    print("=" * 60 + "\n")


def add_remote_agent(
    agent: NixlAgent, sender_meta: NixlAgentMetadata, args: argparse.Namespace
):
    remote_agent_name = agent.add_remote_agent(sender_meta.agent_metadata)
    if isinstance(remote_agent_name, bytes):
        remote_agent_name = remote_agent_name.decode("utf-8")
    remote_xfer_descs = create_xfer_descs(
        agent,
        sender_meta.kv_caches_base_addr[0],
        sender_meta.num_blocks,
        sender_meta.block_len,
        args.nixl_memory_type,
    )
    remote_xfer_handle = agent.prep_xfer_dlist(remote_agent_name, remote_xfer_descs)
    return agent, remote_xfer_handle


def test(
    sender_meta: NixlAgentMetadata,
    agent: NixlAgent,
    local_xfer_handle: str,
    remote_xfer_handle: str,
    local_kv_cache: torch.Tensor,
    args: argparse.Namespace,
    is_warmup: bool = False,
):
    latencies: List[float] = []
    data_per_iteration = args.blocks_per_xfer * sender_meta.block_len
    total_data_transferred = 0
    iteration_history = []
    num_iterations = args.num_iterations if not is_warmup else 1

    logging.info(f"Starting transfer loop for {num_iterations} iterations...")
    logging.info(
        f"Each iteration transfers {args.blocks_per_xfer} blocks "
        f"({data_per_iteration / 1e6:.2f} MB)"
    )

    start_time = time.perf_counter()

    for i in range(num_iterations):
        start = i * args.blocks_per_xfer
        end = start + args.blocks_per_xfer
        block_ids = list(range(start, end))
        iteration_history.append((start, end))

        latency_ms, data_transferred = read_blocks(
            block_ids, agent, local_xfer_handle, remote_xfer_handle, sender_meta
        )

        assert data_transferred == data_per_iteration, (
            f"Data mismatch: expected {data_per_iteration}, got {data_transferred}"
        )
        total_data_transferred += data_transferred
        latencies.append(latency_ms)

    end_time = time.perf_counter()

    # Data integrity checks
    for i, ((start, end), latency_ms) in enumerate(zip(iteration_history, latencies)):
        logging.info(
            f"Iteration {i}: block_ids range is {start} to {end - 1}, "
            f"Latency: {latency_ms:.3f} ms"
        )
        if is_warmup:
            logging.info("Warmup iteration completed, skipping data integrity check.")
            continue

        start_idx = start * sender_meta.block_len // local_kv_cache.element_size()
        end_idx = end * sender_meta.block_len // local_kv_cache.element_size()
        sender_block_hashes = sender_meta.block_hashes[start:end]
        rx_block_hashes = [
            compute_block_hash(local_kv_cache, sender_meta.block_len, j)
            for j in range(start, end)
        ]
        logging.info(
            f"received block is local_kv_cache[{start_idx}:{end_idx}]="
            f"{local_kv_cache[start_idx:end_idx]}"
        )

        if rx_block_hashes != sender_block_hashes:
            logging.error(
                f"DATA INTEGRITY CHECK FAILED for transferred blocks for iteration {i}"
            )
            logging.error(f"Expected: {sender_block_hashes[:2]}...")
            logging.error(f"Received: {rx_block_hashes[:2]}...")
            logging.error("BENCHMARK FAILED - Data corruption detected")
        else:
            logging.info(
                f"Data integrity verified for {args.blocks_per_xfer} "
                f"transferred blocks for iteration {i}"
            )

    total_elapsed_seconds = end_time - start_time
    sustained_throughput_gbps = (total_data_transferred / 1e9) / total_elapsed_seconds

    logging.info("All transfers completed successfully")
    return (
        latencies,
        total_data_transferred,
        sustained_throughput_gbps,
        total_elapsed_seconds,
    )


# ==============================================================================
# Receiver
# ==============================================================================


def receiver_process(args: argparse.Namespace):
    """
    Receiver process: Connects to sender, gets metadata, performs transfers,
    and reports benchmark results.
    """
    logging.info("Receiver starting...")
    agent = None
    config = nixl_agent_config(backends=[args.nixl_backend])

    try:
        time.sleep(5)  # small delay to let sender come up
        receiver_agent_id = str(uuid.uuid4())
        agent = NixlAgent(receiver_agent_id, config)
        logging.info(f"Created receiver agent {receiver_agent_id}")

        with zmq.Context() as ctx, ctx.socket(zmq.REQ) as sock:
            zmq_addr = f"tcp://{args.zmq_host}:{ZMQ_BASE_PORT}"
            sock.connect(zmq_addr)
            logging.info(f"Requesting metadata from sender at {zmq_addr}...")
            sock.send(GET_META_MSG)
            metadata_bytes = sock.recv()

        sender_meta: NixlAgentMetadata = msgspec.msgpack.Decoder(
            NixlAgentMetadata
        ).decode(metadata_bytes)
        logging.info(f"Received metadata from sender engine: {sender_meta.engine_id}")

        agent, remote_xfer_handle = add_remote_agent(agent, sender_meta, args)

        dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
        local_kv_cache = allocate_kv_cache_receiver(
            sender_meta.num_blocks, sender_meta.block_len, dtype, args.device_type
        )

        local_base_addr = get_data_ptr(local_kv_cache, args.device_type)
        reg_descs = agent.get_reg_descs(
            [
                (
                    local_base_addr,
                    local_kv_cache.numel() * local_kv_cache.element_size(),
                    0,
                    "",
                )
            ],
            args.nixl_memory_type,
        )
        agent.register_memory(reg_descs, backends=[args.nixl_backend])

        local_xfer_descs = create_xfer_descs(
            agent,
            local_base_addr,
            sender_meta.num_blocks,
            sender_meta.block_len,
            args.nixl_memory_type,
        )
        local_xfer_handle = agent.prep_xfer_dlist("NIXL_INIT_AGENT", local_xfer_descs)

        logging.info("Starting benchmark test...")
        (
            latencies,
            total_data_transferred,
            sustained_throughput_gbps,
            total_elapsed_seconds,
        ) = test(
            sender_meta,
            agent,
            local_xfer_handle,
            remote_xfer_handle,
            local_kv_cache,
            args,
        )

        summary(
            latencies,
            sender_meta,
            args,
            total_data_transferred,
            sustained_throughput_gbps,
            total_elapsed_seconds,
            agent,
        )

    except Exception:
        logging.error("Receiver process failed", exc_info=True)
    finally:
        # Tell sender to shut down
        try:
            with zmq.Context() as ctx, ctx.socket(zmq.REQ) as sock:
                sock.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
                sock.setsockopt(zmq.SNDTIMEO, 1000)  # 1 second timeout
                zmq_addr = f"tcp://{args.zmq_host}:{ZMQ_BASE_PORT}"
                sock.connect(zmq_addr)
                logging.info("Sending shutdown signal to sender.")
                sock.send(SHUTDOWN_MSG)
                sock.recv()
                logging.info("Sender acknowledged shutdown.")
        except zmq.ZMQError as e:
            logging.info(f"Sender already shutdown or unreachable: {e}")

        logging.info("Receiver shutting down.")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NIXL receiver for KV cache benchmark."
    )
    parser.add_argument("--num-blocks", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=16, help="Tokens per block")
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--blocks-per-xfer", type=int, default=128)
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument(
        "--nixl-memory-type",
        type=str,
        default="DRAM",
        choices=["DRAM", "VRAM"],
    )
    parser.add_argument(
        "--device-type",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "hpu"],
    )
    parser.add_argument("--nixl_backend", type=str, default="OFI")
    parser.add_argument(
        "--ucx-transport",
        type=str,
        default=None,
        help="default is tcp, you might configure as 'cuda_copy,sm'",
    )
    parser.add_argument(
        "--debug-ucx",
        action="store_true",
        help="Enable debug mode for UCX backend (if using UCX backend)",
    )
    parser.add_argument(
        "--zmq-host",
        type=str,
        default="127.0.0.1",
        help="ZMQ host address (use sender's IP or hostname)",
    )

    args = parser.parse_args()

    # UCX env configuration if desired
    if args.debug_ucx:
        os.environ["UCX_PROTO_INFO"] = "y"
    if args.ucx_transport:
        os.environ["UCX_TLS"] = args.ucx_transport
    if args.nixl_backend == "UCX":
        os.environ["UCX_MEMTYPE_CACHE"] = "0"

    # Normalize memory type from device
    if args.device_type == "cpu":
        args.nixl_memory_type = "DRAM"
    elif args.device_type == "cuda":
        args.nixl_memory_type = "VRAM"
    elif args.device_type == "hpu":
        args.nixl_memory_type = "VRAM"  # Gaudi HBM treated as DRAM_SEG

    if args.device_type == "cuda" and not torch.cuda.is_available():
        logging.error("CUDA device specified but not available. Exiting.")
        raise SystemExit(1)

    # Ensure we keep a consistent mapping across blocks & iterations
    args.blocks_per_xfer = min(args.blocks_per_xfer, args.num_blocks)
    if args.blocks_per_xfer * args.num_iterations != args.num_blocks:
        args.num_iterations = args.num_blocks // args.blocks_per_xfer
        logging.warning(
            f"Adjusted num_iterations to {args.num_iterations} to match num_blocks."
        )

    logging.info("Starting receiver...")
    try:
        receiver_process(args)
    except KeyboardInterrupt:
        logging.info("Receiver interrupted by user")

    logging.info("Receiver finished.")
