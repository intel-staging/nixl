#!/usr/bin/env python3
"""
NIXL API Performance Benchmark - Sender

Allocates KV cache, registers memory with NIXL, computes block hashes, and
serves metadata to the receiver over ZMQ. Designed to be run standalone on
the sender node.
"""

import argparse
import logging
import os
import time
import socket
import uuid
import hashlib
from typing import List


import msgspec
import torch
import zmq

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


def detect_node_ip() -> str:
    """
    Try to detect a non-loopback IPv4 address for this node.

    Strategy:
    1. Create a UDP socket and "connect" to a public IP (no traffic needed),
       then read the local address used for that route.
    2. Fallback to gethostbyname(gethostname()).
    3. Final fallback: 127.0.0.1
    """
    # Prefer something routable (e.g., for other nodes)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # This doesn't actually send packets; it's just for route selection
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
        finally:
            s.close()
    except OSError:
        pass

    # Fallback to hostname-based resolution
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip:
            return ip
    except OSError:
        pass

    # Last resort
    return "127.0.0.1"


def get_data_ptr(t: torch.Tensor, device_type: str = "cpu") -> int:
    if device_type == "hpu":
        addr = htexp._data_ptr(t)
    else:
        addr = t.data_ptr()
    logging.info(f"Data ptr of tensor on device {t.device}: 0x{addr:x}")
    return addr


def allocate_kv_cache(
    num_blocks: int, block_len: int, dtype: torch.dtype, device: str
) -> torch.Tensor:
    """Allocate a KV cache buffer on the given device."""
    total_bytes = num_blocks * block_len
    num_elements = total_bytes // dtype.itemsize
    logging.info(
        f"Allocating KV cache: {total_bytes / 1e6:.2f} MB "
        f"({num_elements:,} elements, {dtype}, {device})"
    )
    # initialize a random tensor to avoid all-zero data
    return torch.rand(num_elements, dtype=dtype, device=device)


def compute_block_hash(kv_cache: torch.Tensor, block_len: int, block_idx: int) -> str:
    elem_per_blk = block_len // kv_cache.element_size()
    start = block_idx * elem_per_blk
    end = start + elem_per_blk
    block_bytes = kv_cache[start:end].cpu().numpy().tobytes()
    return hashlib.sha256(block_bytes).hexdigest()


def create_xfer_descs(
    agent: NixlAgent, base_addr: int, num_blocks: int, block_len: int, mem_type: str
):
    """Create transfer descriptors for a block range."""
    blocks_data = [(base_addr + i * block_len, block_len, 0) for i in range(num_blocks)]
    return agent.get_xfer_descs(blocks_data, mem_type)


# ==============================================================================
# Sender
# ==============================================================================


def sender_process(args: argparse.Namespace):
    """
    Sender process: Allocates memory, registers it with NIXL,
    and waits to serve metadata and a shutdown signal.
    """
    logging.info("Sender starting...")
    node_ip = detect_node_ip()
    logging.info(f"Detected node IP address: {node_ip}")
    logging.info(
        f"Receivers can connect with: --zmq-host {node_ip} "
        f"(ZMQ endpoint tcp://{node_ip}:{ZMQ_BASE_PORT})"
    )
    agent = None
    config = nixl_agent_config(backends=[args.nixl_backend])

    try:
        sender_agent_id = str(uuid.uuid4())
        agent = NixlAgent(sender_agent_id, config)

        dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
        block_len = (
            args.num_heads * args.head_size * 2 * args.block_size * dtype.itemsize
        )

        # Prepare the KV cache on sender
        kv_cache = allocate_kv_cache(
            args.num_blocks, block_len, dtype, args.device_type
        )

        # Some diagnostics
        print(f"=== KV Cache Memory Analysis ===")
        print(f"KV cache device: {kv_cache.device}")
        print(f"KV cache dtype: {kv_cache.dtype}")
        print(f"KV cache data_ptr: 0x{kv_cache.data_ptr():x}")
        print(f"KV cache size: {kv_cache.numel() * kv_cache.element_size()} bytes")

        if hasattr(kv_cache, "device_type"):
            print(f"Device type: {kv_cache.device_type}")
        if hasattr(kv_cache, "is_cuda"):
            print(f"Is CUDA memory: {kv_cache.is_cuda}")

        try:
            import habana_frameworks.torch as ht

            hpu_available = False
            if hasattr(ht, "is_available"):
                hpu_available = ht.is_available()
            elif hasattr(ht, "hpu_is_available"):
                hpu_available = ht.hpu_is_available()
            else:
                try:
                    test_tensor = ht.tensor([1.0], device="hpu")
                    hpu_available = True
                    del test_tensor
                except Exception:
                    hpu_available = False

            print(f"Habana frameworks available: {hpu_available}")
            if hpu_available:
                if hasattr(kv_cache, "is_hpu"):
                    print(f"Is HPU memory: {kv_cache.is_hpu}")
                else:
                    device_str = str(kv_cache.device).lower()
                    print(f"Device string: {device_str}")
                    print(f"Likely HPU memory: {'hpu' in device_str}")

        except ImportError as e:
            print(f"Habana frameworks not available: {e}")
        except Exception as e:
            print(f"Error checking Habana framework: {e}")

        if hasattr(kv_cache, "storage"):
            storage = kv_cache.storage()
            print(f"Storage device: {getattr(storage, 'device', 'Unknown')}")
        print(f"==================================")

        # Compute SHA-256 hashes for each block
        logging.info("Computing block hashes for data integrity verification...")
        block_hashes: List[str] = [
            compute_block_hash(kv_cache, block_len, i) for i in range(args.num_blocks)
        ]

        # Register memory with NIXL
        base_addr = get_data_ptr(kv_cache, args.device_type)
        reg_descs = agent.get_reg_descs(
            [(base_addr, kv_cache.numel() * kv_cache.element_size(), 0, "")],
            args.nixl_memory_type,
        )
        agent.register_memory(reg_descs, backends=[args.nixl_backend])

        local_xfer_descs = create_xfer_descs(
            agent, base_addr, args.num_blocks, block_len, args.nixl_memory_type
        )
        agent.prep_xfer_dlist("NIXL_INIT_AGENT", local_xfer_descs)

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

        # ZMQ server (ROUTER) for metadata + shutdown handshake
        with zmq.Context() as ctx, ctx.socket(zmq.ROUTER) as sock:
            zmq_addr = f"tcp://*:{ZMQ_BASE_PORT}"
            sock.bind(zmq_addr)
            logging.info(f"Sender listening for handshakes on {zmq_addr}")

            # 1) Wait for metadata request
            identity, _, msg = sock.recv_multipart()
            if msg == GET_META_MSG:
                sock.send_multipart((identity, b"", encoded_metadata))
                logging.info("Sent metadata to receiver.")
            else:
                raise RuntimeError(f"Expected metadata request, got: {msg}")

            # 2) Wait for shutdown signal
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


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIXL sender for KV cache benchmark.")
    parser.add_argument("--num-blocks", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=16, help="Tokens per block")
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    # blocks-per-xfer / num-iterations are not used by sender, but keeping them
    # for CLI symmetry with receiver.
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

    logging.info("Starting sender...")
    try:
        sender_process(args)
    except KeyboardInterrupt:
        logging.info("Sender interrupted by user")

    logging.info("Sender finished.")
