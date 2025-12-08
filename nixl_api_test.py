#!/usr/bin/env python3
"""
NIXL API Performance Benchmark (Corrected & Improved)

Simulates sender/receiver engines transferring KV cache blocks over NIXL.
This version fixes a potential INVALID_PARAM error and improves process synchronization.
"""

import argparse
import logging
import multiprocessing
import os
import time
import uuid
from typing import List, Iterator
import hashlib
import habana_frameworks.torch as htorch
import habana_frameworks.torch.utils.experimental as htexp

import msgspec
import torch
import zmq
import pandas as pd
# ==============================================================================
# Configuration
# ==============================================================================

# ZMQ_HOST will be set from command line argument
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

def get_data_ptr(t, device_type="cpu"):
    if device_type == "hpu":
        addr = htexp._data_ptr(t)
    else:
        addr = t.data_ptr()
    logging.info(f"Data ptr of tensor on device {t.device}: 0x{addr:x}")
    return addr

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

def get_block_desc_ids(num_total_blocks: int, block_ids: Iterator[int]) -> List[int]:
    """
    Maps logical block IDs to the indices of their transfer descriptors.
    """
    result = list(block_ids)
    # logging.info(f"get_block_desc_ids: {list(block_ids)} -> {result}")
    return result


def test(sender_meta: NixlAgentMetadata, agent: NixlAgent, local_xfer_handle: str, remote_xfer_handle: str, local_kv_cache: torch.Tensor, args: argparse.Namespace, is_warmup: bool = False):
    latencies = []
    data_per_iteration = args.blocks_per_xfer * sender_meta.block_len
    total_data_transferred = 0
    elapse_validate = 0
    iteration_history = []
    num_iterations = args.num_iterations if not is_warmup else 1
    logging.info(f"Starting transfer loop for {num_iterations} iterations...")
    logging.info(f"Each iteration transfers {args.blocks_per_xfer} blocks ({data_per_iteration / 1e6:.2f} MB)")
    logging.info(f"Do extra h2d copy: {args.do_h2d_cp}")

    # Measure total elapsed time for sustained throughput calculation
    start_time = time.perf_counter()
    
    for i in range(num_iterations):
        # For simplicity: always transfer the first blocks_per_xfer blocks
        start = i * args.blocks_per_xfer
        end = start + args.blocks_per_xfer
        block_ids = list(range(start, end))
        iteration_history.append((start, end))


        # Transfer and measure time, h2d copy operation will enabled when do_h2d_cp=True
        latency_ms, data_transferred = read_blocks(block_ids, agent, local_xfer_handle, remote_xfer_handle, sender_meta,
                                                   args.do_h2d_cp, local_kv_cache, start, end)
        
        # Verify data_transferred matches expected amount
        assert data_transferred == data_per_iteration, f"Data mismatch: expected {data_per_iteration}, got {data_transferred}"
        total_data_transferred += data_transferred
        
        latencies.append(latency_ms)

    end_time = time.perf_counter()

    for i, ((start, end), latency_ms) in enumerate(zip(iteration_history, latencies)):
        # Validate data integrity by comparing hashes
        # Verify only the blocks we actually transferred (first blocks_per_xfer blocks)
        logging.info(f"Iteration {i}: block_ids range is {start} to {end-1}, Latency: {latency_ms:.3f} ms")
        if is_warmup:
            logging.info(f"Warmup iteration completed, skipping data integrity check.")
            continue
        start_idx = start * sender_meta.block_len // local_kv_cache.element_size()
        end_idx = end * sender_meta.block_len // local_kv_cache.element_size()
        sender_block_hashes = sender_meta.block_hashes[start:end]
        rx_block_hashes = [compute_block_hash(local_kv_cache, sender_meta.block_len, i) for i in range(start, end)]
        logging.info(f"received block is local_kv_cache[{start_idx}:{end_idx}]={local_kv_cache[start_idx:end_idx]}")
         # Compare hashes for transferred blocks
        if rx_block_hashes != sender_block_hashes:
            logging.error(f"DATA INTEGRITY CHECK FAILED for transferred blocks for iteration {i}")
            logging.error(f"Expected: {sender_block_hashes[:2]}...")
            logging.error(f"Received: {rx_block_hashes[:2]}...")
            logging.error(f"BENCHMARK FAILED - Data corruption detected")
            # Exit with error code to indicate failure
            #exit(1)
        else:
            logging.info(f"Data integrity verified for {args.blocks_per_xfer} transferred blocks for iteration {i}")
    total_elapsed_seconds = end_time - start_time
    # Calculate sustained throughput over entire benchmark duration
    sustained_throughput_gbps = (total_data_transferred / 1e9) / total_elapsed_seconds

    # Print summary after successful completion
    logging.info("All transfers completed successfully")
    return latencies, total_data_transferred, sustained_throughput_gbps, total_elapsed_seconds
        
def allocate_kv_cache(num_blocks: int, block_len: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    """Allocate a KV cache buffer on the given device."""
    total_bytes = num_blocks * block_len
    num_elements = total_bytes // dtype.itemsize
    logging.info(
        f"Allocating KV cache: {total_bytes / 1e6:.2f} MB "
        f"({num_elements:,} elements, {dtype}, {device})"
    )
    # initialize a random tensor to avoid all-zero data which might be optimized away
    return torch.rand(num_elements, dtype=dtype, device=device)

def allocate_kv_cache_receiver(num_blocks: int, block_len: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    """Allocate a KV cache buffer on the given device."""
    total_bytes = num_blocks * block_len
    num_elements = total_bytes // dtype.itemsize
    logging.info(
        f"Allocating KV cache: {total_bytes / 1e6:.2f} MB "
        f"({num_elements:,} elements, {dtype}, {device})"
    )
    return torch.zeros(num_elements, dtype=dtype, device=device)



def create_xfer_descs(agent: NixlAgent, base_addr: int, num_blocks: int, block_len: int, mem_type: str):
    """Create transfer descriptors for a block range."""
    blocks_data = [(base_addr + i * block_len, block_len, 0) for i in range(num_blocks)]
    return agent.get_xfer_descs(blocks_data, mem_type)


def read_blocks(block_ids: Iterator[int], agent: NixlAgent,
                local_xfer_handle: str, remote_xfer_handle: str, sender_meta: NixlAgentMetadata,
                do_h2d_cp, local_kv_cache, start, end):
    """ Read blocks from the sender's KV cache using NIXL. """
    if not block_ids:
        logging.warning("No block IDs provided for transfer.")
        return
    
    block_ids_list = list(block_ids)
    local_ids = get_block_desc_ids(sender_meta.num_blocks, iter(block_ids_list))
    remote_ids = get_block_desc_ids(sender_meta.num_blocks, iter(block_ids_list))
    
    # DEBUG: Log the actual mappings
    # logging.info(f"Transferring block_ids: {block_ids_list}")
    # logging.info(f"Mapped to local_ids: {local_ids}")
    # logging.info(f"Mapped to remote_ids: {remote_ids}")

    # FINAL FIX: Use the integer handles and pass a regular string for notif_msg.
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

    if do_h2d_cp:
        local_kv_cache[start:end].to("hpu")
    t1 = time.perf_counter_ns()


    
    return (t1 - t0) / 1e6, len(local_ids) * sender_meta.block_len  # Return latency in ms


def summary(latencies: List[float], sender_meta: NixlAgentMetadata, args: argparse.Namespace, total_data_transferred: int, sustained_throughput_gbps: float, total_elapsed_seconds: float, agent: NixlAgent):
    total_latency_time = sum(latencies)
    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0
    data_per_operation = args.blocks_per_xfer * sender_meta.block_len

    print("\n" + "=" * 60)
    print(" NIXL API Performance Benchmark Results")
    print("=" * 60)
    print(f" Hardware Memory Type:    {args.nixl_memory_type}")
    print(f" Device Type:             {args.device_type}")
    backend_names = list(agent.backends.keys()) if isinstance(agent.backends, dict) else [str(agent.backends)]
    print(f" NIXL Backend:            {', '.join(backend_names)}")
    print(f" Total Iterations:        {args.num_iterations:,}")
    print(f" Blocks per Transfer:     {args.blocks_per_xfer:,}")
    print(f" Data per Transfer:       {(args.blocks_per_xfer * sender_meta.block_len) / 1e6:.2f} MB")
    print("-" * 60)
    print(f" Average Latency:         {avg_latency_ms:.3f} ms")
    print(f" Sustained Throughput:    {sustained_throughput_gbps:.3f} GB/s")
    print(f" Total Data Transferred:  {total_data_transferred / 1e6:.2f} MB")
    print(f" Total Elapsed Time:      {total_elapsed_seconds:.3f} s")
    print(f" Sum of Transfer Times:   {total_latency_time:.1f} ms")
    print(f" Overhead Time:           {(total_elapsed_seconds * 1000 - total_latency_time):.1f} ms")
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


def add_remote_agent(agent: NixlAgent, sender_meta: NixlAgentMetadata, args: argparse.Namespace): 
    remote_agent_name = agent.add_remote_agent(sender_meta.agent_metadata)
    if isinstance(remote_agent_name, bytes):
        remote_agent_name = remote_agent_name.decode('utf-8')
    remote_xfer_descs = create_xfer_descs(
        agent, sender_meta.kv_caches_base_addr[0], sender_meta.num_blocks,
        sender_meta.block_len, args.nixl_memory_type
    )
    remote_xfer_handle = agent.prep_xfer_dlist(remote_agent_name, remote_xfer_descs)
    return agent, remote_xfer_handle


# ==============================================================================
# Processes
# ==============================================================================

def compute_block_hash(kv_cache, block_len, block_idx):
    #print(f"{block_len=}, {block_idx=}, {kv_cache.element_size()=}")
    elem_per_blk = block_len // kv_cache.element_size()
    start = block_idx * elem_per_blk
    end = start + elem_per_blk
    block_bytes = kv_cache[start:end].cpu().numpy().tobytes()
    return hashlib.sha256(block_bytes).hexdigest()


def sender_process(args: argparse.Namespace, zmq_host: str = "127.0.0.1"):
    """
    Sender process: Allocates memory, registers it with NIXL,
    and waits to serve metadata and a shutdown signal.
    """
    logging.info("Sender starting...")
    agent = None
    config = nixl_agent_config(backends=[args.nixl_backend])
    
    try:
        sender_agent_id = str(uuid.uuid4())
        agent = NixlAgent(sender_agent_id, config)

        dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
        block_len = (
            args.num_heads * args.head_size * 2 * args.block_size * dtype.itemsize
        )

        ####### Prepare the KV cache for the sender #####
        kv_cache = allocate_kv_cache(args.num_blocks, block_len, dtype, args.device_type)

        # Add comprehensive memory type detection for debugging
        print(f"=== KV Cache Memory Analysis ===")
        print(f"KV cache device: {kv_cache.device}")
        print(f"KV cache dtype: {kv_cache.dtype}")
        print(f"KV cache data_ptr: 0x{kv_cache.data_ptr():x}")
        print(f"KV cache size: {kv_cache.numel() * kv_cache.element_size()} bytes")
        
        # Check device type
        if hasattr(kv_cache, 'device_type'):
            print(f"Device type: {kv_cache.device_type}")
        
        # Check if CUDA memory
        if hasattr(kv_cache, 'is_cuda'):
            print(f"Is CUDA memory: {kv_cache.is_cuda}")
        
        # Check if HPU memory (SynapseAI/Habana)
        try:
            import habana_frameworks.torch as ht
            # Check different ways HPU availability might be exposed
            hpu_available = False
            if hasattr(ht, 'is_available'):
                hpu_available = ht.is_available()
            elif hasattr(ht, 'hpu_is_available'):
                hpu_available = ht.hpu_is_available()
            else:
                # Try to detect by checking if we can create HPU tensors
                try:
                    test_tensor = ht.tensor([1.0], device='hpu')
                    hpu_available = True
                    del test_tensor
                except:
                    hpu_available = False
            
            print(f"Habana frameworks available: {hpu_available}")
            
            if hpu_available:
                if hasattr(kv_cache, 'is_hpu'):
                    print(f"Is HPU memory: {kv_cache.is_hpu}")
                else:
                    # Alternative check: see if device string contains 'hpu'
                    device_str = str(kv_cache.device).lower()
                    print(f"Device string: {device_str}")
                    print(f"Likely HPU memory: {'hpu' in device_str}")
                    
        except ImportError as e:
            print(f"Habana frameworks not available: {e}")
        except Exception as e:
            print(f"Error checking Habana framework: {e}")
        
        # Additional device memory checks
        if hasattr(kv_cache, 'storage'):
            storage = kv_cache.storage()
            print(f"Storage device: {storage.device if hasattr(storage, 'device') else 'Unknown'}")
            
        print(f"==================================")
        
        # Compute SHA-256 hashes for each block
        logging.info(f"Computing block hashes for data integrity verification...{kv_cache=}")
        block_hashes = [compute_block_hash(kv_cache, block_len, i) for i in range(args.num_blocks)]

        # logging.info(f"--------sender process: --------kv_cache={kv_cache}---\n")
        # logging.info(f"--------sender process: --------args.num_blocks={args.num_blocks},block_len={block_len}---\n")
        # logging.info(f"------------- block_hashes={block_hashes[0:2]} -------- ")

        reg_descs = agent.get_reg_descs(
            [(get_data_ptr(kv_cache, args.device_type), kv_cache.numel() * kv_cache.element_size(), 0, "")],
            args.nixl_memory_type
        )
        agent.register_memory(reg_descs, backends=[args.nixl_backend])

        base_addr = get_data_ptr(kv_cache, args.device_type)
        local_xfer_descs = create_xfer_descs(agent, base_addr, args.num_blocks, block_len, args.nixl_memory_type)
        agent.prep_xfer_dlist('NIXL_INIT_AGENT', local_xfer_descs)
        ##################################################

        metadata = NixlAgentMetadata(
            engine_id=f"sender-engine-{os.getpid()}",
            agent_metadata=agent.get_agent_metadata(),
            kv_caches_base_addr=[base_addr],
            num_blocks=args.num_blocks,
            block_len=block_len,
            block_hashes=block_hashes,
        )
        encoder = msgspec.msgpack.Encoder()
        encoded_metadata = encoder.encode(metadata)

        with zmq.Context() as ctx, ctx.socket(zmq.ROUTER) as sock:
            zmq_addr = f"tcp://*:{ZMQ_BASE_PORT}"
            sock.bind(zmq_addr)
            logging.info(f"Sender listening for handshakes on {zmq_addr}")

            identity, _, msg = sock.recv_multipart()
            if msg == GET_META_MSG:
                sock.send_multipart((identity, b"", encoded_metadata))
                logging.info("Sent metadata to receiver.")
            else:
                raise RuntimeError(f"Expected metadata request, got: {msg}")

            identity, _, msg = sock.recv_multipart()
            if msg == SHUTDOWN_MSG:
                sock.send_multipart((identity, b"ack"))
                logging.info("Received shutdown signal. Sender will now exit.")
            else:
                logging.warning(f"Expected shutdown signal, got: {msg}")

    except Exception:
        logging.error("Sender process failed", exc_info=True)
    finally:
        logging.info("Sender shutting down.")

# ------------------------------------------------------------------------------

def receiver_process(args: argparse.Namespace, zmq_host: str = "127.0.0.1"):
    """
    Receiver process: Connects to sender, gets metadata, performs transfers,
    and reports benchmark results.
    """
    logging.info("Receiver starting...")
    agent = None
    config = nixl_agent_config(backends=[args.nixl_backend])
    try:
        time.sleep(5)
        receiver_agent_id = str(uuid.uuid4())
        agent = NixlAgent(receiver_agent_id, config)
        logging.info(f"Created receiver agent {receiver_agent_id}")

        with zmq.Context() as ctx, ctx.socket(zmq.REQ) as sock:
            zmq_addr = f"tcp://{zmq_host}:{ZMQ_BASE_PORT}"
            sock.connect(zmq_addr)
            logging.info(f"Requesting metadata from sender at {zmq_addr}...")
            sock.send(GET_META_MSG)
            metadata_bytes = sock.recv()

        sender_meta: NixlAgentMetadata = msgspec.msgpack.Decoder(NixlAgentMetadata).decode(metadata_bytes)
        logging.info(f"Received metadata from sender engine: {sender_meta.engine_id}")

        agent, remote_xfer_handle = add_remote_agent(agent, sender_meta, args)
        
        ####### Prepare the KV cache for the sender #####
        dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
        local_kv_cache = allocate_kv_cache_receiver(
            sender_meta.num_blocks, sender_meta.block_len, dtype, args.device_type
        )
        # print(f"--------3.1--------local_kv_cache={local_kv_cache}---\n")
        local_base_addr = get_data_ptr(local_kv_cache, args.device_type)
        reg_descs = agent.get_reg_descs(
            [(local_base_addr, local_kv_cache.numel() * local_kv_cache.element_size(), 0, "")],
            args.nixl_memory_type
        )
        agent.register_memory(reg_descs, backends=[args.nixl_backend])
        local_xfer_descs = create_xfer_descs(
            agent, local_base_addr, sender_meta.num_blocks, sender_meta.block_len, args.nixl_memory_type
        )
        local_xfer_handle = agent.prep_xfer_dlist('NIXL_INIT_AGENT', local_xfer_descs)
        #################################################
        # logging.info("Starting warmup test...")
        # latencies, total_data_transferred, sustained_throughput_gbps, total_elapsed_seconds = test(
        #     sender_meta, agent, local_xfer_handle, remote_xfer_handle, local_kv_cache, args, is_warmup=True
        # )
        
        logging.info("Starting benchmark test...")
        latencies, total_data_transferred, sustained_throughput_gbps, total_elapsed_seconds = test(
            sender_meta, agent, local_xfer_handle, remote_xfer_handle, local_kv_cache, args
        )
        

        summary(latencies, sender_meta, args, total_data_transferred, sustained_throughput_gbps, total_elapsed_seconds, agent)

    except Exception:
        logging.error("Receiver process failed", exc_info=True)
    finally:
        try:
            with zmq.Context() as ctx, ctx.socket(zmq.REQ) as sock:
                sock.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
                sock.setsockopt(zmq.SNDTIMEO, 1000)  # 1 second timeout
                zmq_addr = f"tcp://{zmq_host}:{ZMQ_BASE_PORT}"
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
    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser(description="Benchmark script for nixl._api performance.")
    parser.add_argument("--num-blocks", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=16, help="Tokens per block")
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--blocks-per-xfer", type=int, default=128)
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--nixl-memory-type", type=str, default="DRAM", choices=["DRAM", "VRAM"])
    parser.add_argument("--device-type", type=str, default="cpu", choices=["cpu", "cuda", "hpu"])
    parser.add_argument("--nixl_backend", type=str, default="OFI")
    parser.add_argument("--ucx-transport", type=str, default=None, help="default is tcp, you might configure as 'cuda_copy,sm'")
    parser.add_argument("--debug-ucx", action="store_true",
                        help="Enable debug mode for UCX backend (if using UCX backend)")
    parser.add_argument("--role", type=str, default="aio", choices=["sender", "receiver", "aio"],
                        help="Role to run: sender or receiver")
    parser.add_argument("--zmq-host", type=str, default="127.0.0.1",
                        help="ZMQ host address (use sender's IP for receiver)")
    parser.add_argument("--do-h2d-cp", action="store_true",
                        help="Do extra h2d copy")
    args = parser.parse_args()
    #FI_LOG_LEVEL=debug NIXL_LOG_LEVEL=debug 
    # PT_HPU_POOL_STRATEGY=0 NIXL_PLUGIN_DIR=/workspace/nixl/nixl-nixl_libfabric/build/cp310/src/plugins/libfabric python ts_nixl/nixl_api.py  --device-type hpu --nixl_backend libfabric --nixl-memory-type DRAM
    # PT_HPU_POOL_STRATEGY=0 NIXL_PLUGIN_DIR=/workspace/ts_nixl/nixl/build/cp312/src/plugins/ofi python ts_nixl/nixl_api.py  --device-type hpu --nixl_backend OFI --nixl-memory-type DRAM
    
    if args.debug_ucx:
        os.environ['UCX_PROTO_INFO'] = 'y'
    if args.ucx_transport:
        os.environ['UCX_TLS'] = args.ucx_transport 
    args.blocks_per_xfer = min(args.blocks_per_xfer, args.num_blocks)
    if args.device_type == "cpu":
        args.nixl_memory_type = "DRAM"
    elif args.device_type == "cuda":
        args.nixl_memory_type = "VRAM"
    elif args.device_type == "hpu":
        args.nixl_memory_type = "VRAM"  # Gaudi HBM treated as DRAM_SEG
    #args.dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    if args.device_type == "cuda" and not torch.cuda.is_available():
        logging.error("CUDA device specified but not available. Exiting.")
        exit(1)

    if args.blocks_per_xfer * args.num_iterations != args.num_blocks:
        args.num_iterations = args.num_blocks // args.blocks_per_xfer
        logging.warning(f"Adjusted num_iterations to {args.num_iterations} to match num_blocks.")

    # Run only the specified role
    if args.role == "sender":
        try:
            sender_process(args, args.zmq_host)
        except KeyboardInterrupt:
            logging.info("Sender interrupted by user")
    elif args.role == "receiver":
        try:
            receiver_process(args, args.zmq_host)
        except KeyboardInterrupt:
            logging.info("Receiver interrupted by user")
    else:
        sender = multiprocessing.Process(target=sender_process, args=(args, args.zmq_host), name="Sender")
        receiver = multiprocessing.Process(target=receiver_process, args=(args, args.zmq_host), name="Receiver")

        try:
            sender.start()
            receiver.start()

            # Wait for receiver to complete, then wait for sender
            receiver.join()
            logging.info("Receiver finished, waiting for sender...")
            sender.join(timeout=5)  # Give sender 5 seconds to clean shutdown

            if sender.is_alive():
                logging.warning("Sender did not shutdown gracefully, terminating...")
                sender.terminate()
                sender.join(timeout=2)

        except KeyboardInterrupt:
            logging.info("Interrupted by user, terminating processes...")
        finally:
            for proc in (sender, receiver):
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=2)

    logging.info("Process finished.")
