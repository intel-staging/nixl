# install_prerequisites.py
import os
import subprocess
import sys
import argparse
import glob
import shutil

# --- Configuration ---
WHEELS_CACHE_HOME = os.environ.get("WHEELS_CACHE_HOME", "/tmp/wheels_cache")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
UCX_DIR = os.path.join("/tmp", "ucx_source")
LIBFABRIC_DIR = os.path.join("/tmp", "libfabric_source")
NIXL_DIR = os.path.join("/tmp", "nixl_source")
UCX_INSTALL_DIR = os.path.join("/tmp", "ucx_install")
LIBFABRIC_INSTALL_DIR = os.path.join("/tmp", "libfabric_install")

# --- Repository and Version Configuration ---
UCX_REPO_URL = "https://github.com/intel-staging/ucx.git"
UCX_BRANCH = "intel_gaudi_gdr_enabling_0"
LIBFABRIC_REPO_URL = "https://github.com/ofiwg/libfabric.git"
LIBFABRIC_REF = "v1.21.0"  # Using a recent stable tag
# TODO: Change back nixl repo to an intel-staging branch
NIXL_REPO_URL = "https://github.com/pi314ever/nixl.git"
NIXL_BRANCH = "v0.6.0_UCX"


# --- Helper Functions ---
def run_command(command, cwd=".", env=None):
    """Helper function to run a shell command and check for errors."""
    print(f"--> Running command: {' '.join(command)} in '{cwd}'", flush=True)
    subprocess.check_call(command, cwd=cwd, env=env)


def is_pip_package_installed(package_name):
    """Checks if a package is installed via pip without raising an exception."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", package_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def find_nixl_wheel_in_cache(cache_dir):
    """Finds a nixl wheel file in the specified cache directory."""
    # The repaired wheel will have a 'manylinux' tag, but this glob still works.
    search_pattern = os.path.join(cache_dir, "nixl-*.whl")
    wheels = glob.glob(search_pattern)
    if wheels:
        # Sort to get the most recent/highest version if multiple exist
        wheels.sort()
        return wheels[-1]
    return None


def install_system_dependencies():
    """Installs required system packages using apt-get if run as root."""
    if os.geteuid() != 0:
        print("\n---", flush=True)
        print(
            "WARNING: Not running as root. Skipping system dependency installation.",
            flush=True,
        )
        print(
            "Please ensure the following packages are installed on your system:",
            flush=True,
        )
        print(
            "  patchelf build-essential git cmake ninja-build autotools-dev automake meson libtool libtool-bin",
            flush=True,
        )
        print("---\n", flush=True)
        return

    print("--- Running as root. Installing system dependencies... ---", flush=True)
    apt_packages = [
        "patchelf",  # <-- Add patchelf here
        "build-essential",
        "git",
        "cmake",
        "ninja-build",
        "autotools-dev",
        "automake",
        "meson",
        "libtool",
        "libtool-bin",
        "libhwloc-dev",
        "zip",
    ]
    run_command(["apt-get", "update"])
    run_command(["apt-get", "install", "-y"] + apt_packages)
    print("--- System dependencies installed successfully. ---\n", flush=True)


def build_and_install_prerequisites(args):
    """Builds UCX and NIXL from source, creating a self-contained wheel."""

    # ... (initial checks and setup are unchanged) ...
    if not args.force_reinstall and is_pip_package_installed("nixl"):
        print("--> NIXL is already installed. Nothing to do.", flush=True)
        return

    cached_wheel = find_nixl_wheel_in_cache(WHEELS_CACHE_HOME)
    if not args.force_reinstall and cached_wheel:
        print(
            f"\n--> Found self-contained wheel: {os.path.basename(cached_wheel)}.",
            flush=True,
        )
        print("--> Installing from cache, skipping all source builds.", flush=True)
        install_command = [sys.executable, "-m", "pip", "install", cached_wheel]
        run_command(install_command)
        print("\n--- Installation from cache complete. ---", flush=True)
        return

    print(
        "\n--> No installed package or cached wheel found. Starting full build process...",
        flush=True,
    )
    print("\n--> Installing auditwheel...", flush=True)
    run_command([sys.executable, "-m", "pip", "install", "auditwheel"])
    install_system_dependencies()
    ucx_install_path = os.path.abspath(UCX_INSTALL_DIR)
    print(f"--> Using wheel cache directory: {WHEELS_CACHE_HOME}", flush=True)
    os.makedirs(WHEELS_CACHE_HOME, exist_ok=True)

    # -- Step 1: Build UCX from source --
    print("\n[1/3] Configuring and building UCX from source...", flush=True)
    if not os.path.exists(UCX_DIR):
        run_command(["git", "clone", UCX_REPO_URL, UCX_DIR])
    ucx_source_path = os.path.abspath(UCX_DIR)
    run_command(["git", "checkout", "v1.19.x"], cwd=ucx_source_path)
    run_command(["./autogen.sh"], cwd=ucx_source_path)
    configure_command = [
        "./configure",
        f"--prefix={ucx_install_path}",
        "--enable-shared",
        "--disable-static",
        "--disable-doxygen-doc",
        "--enable-optimizations",
        "--enable-cma",
        "--enable-devel-headers",
        "--with-verbs",
        "--enable-mt",
    ]
    run_command(configure_command, cwd=ucx_source_path)
    run_command(["make", "-j", str(os.cpu_count() or 1)], cwd=ucx_source_path)
    run_command(["make", "install"], cwd=ucx_source_path)
    print("--- UCX build and install complete ---", flush=True)

    # -- Step 2: Build Libfabric from source --
    print(
        f"\n[2/4] Configuring and building Libfabric (ref: {LIBFABRIC_REF}) from source...",
        flush=True,
    )
    if not os.path.exists(LIBFABRIC_DIR):
        run_command(["git", "clone", LIBFABRIC_REPO_URL, LIBFABRIC_DIR])
    run_command(["git", "checkout", LIBFABRIC_REF], cwd=LIBFABRIC_DIR)
    run_command(["./autogen.sh"], cwd=LIBFABRIC_DIR)
    configure_command_lf = [
        "./configure",
        f"--prefix={LIBFABRIC_INSTALL_DIR}",
        "--enable-verbs",
        "--enable-shm",
        "--enable-sockets",
        "--enable-tcp",
        "--with-synapseai=/usr/include/habanalabs",  # As requested
    ]
    run_command(configure_command_lf, cwd=LIBFABRIC_DIR)
    run_command(["make", "-j", str(os.cpu_count() or 1)], cwd=LIBFABRIC_DIR)
    run_command(["make", "install"], cwd=LIBFABRIC_DIR)
    print("--- Libfabric build and install complete ---", flush=True)

    # -- Step 3: Build NIXL wheel from source --
    print(
        f"\n[3/4] Building NIXL (branch: {NIXL_BRANCH}) wheel from source...",
        flush=True,
    )
    if not os.path.exists(NIXL_DIR):
        run_command(["git", "clone", "--branch", NIXL_BRANCH, NIXL_REPO_URL, NIXL_DIR])

    build_env = os.environ.copy()
    # Configure environment to find both UCX and Libfabric
    ucx_install_path = os.path.abspath(UCX_INSTALL_DIR)
    lf_install_path = os.path.abspath(LIBFABRIC_INSTALL_DIR)

    ucx_pkg_path = os.path.join(ucx_install_path, "lib", "pkgconfig")
    lf_pkg_path = os.path.join(lf_install_path, "lib", "pkgconfig")
    build_env["PKG_CONFIG_PATH"] = f"{ucx_pkg_path}:{lf_pkg_path}".strip(":")

    ucx_lib_path = os.path.join(ucx_install_path, "lib")
    ucx_plugin_path = os.path.join(ucx_lib_path, "ucx")
    lf_lib_path = os.path.join(lf_install_path, "lib")
    build_env["LD_LIBRARY_PATH"] = (
        f"{ucx_lib_path}:{ucx_plugin_path}:{lf_lib_path}".strip(":")
    )

    print(f"--> Using PKG_CONFIG_PATH: {build_env['PKG_CONFIG_PATH']}", flush=True)
    print(f"--> Using LD_LIBRARY_PATH: {build_env['LD_LIBRARY_PATH']}", flush=True)

    temp_wheel_dir = os.path.join(ROOT_DIR, "temp_wheelhouse")
    # Define the build command for nixl wheel with specific meson arguments
    wheel_build_cmd = [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        ".",
        "--no-deps",
        f"--wheel-dir={temp_wheel_dir}",
        # Pass meson arguments via pip's config-settings
        "--config-settings=setup-args=-Ddisable_gds_backend=true",
        f"--config-settings=setup-args=-Dlibfabric_path={lf_install_path}",
        f"--config-settings=setup-args=-Ducx_path={ucx_install_path}",
    ]

    run_command(wheel_build_cmd, cwd=os.path.abspath(NIXL_DIR), env=build_env)

    # -- Step 4: Repair wheel, then replace libfabric --
    # auditwheel may bundle an incompatible libfabric, so we need to replace it
    print(
        "\n[4/4] Repairing wheel with auditwheel and correcting libfabric...",
        flush=True,
    )
    unrepaired_wheel = find_nixl_wheel_in_cache(temp_wheel_dir)
    if not unrepaired_wheel:
        raise RuntimeError("Failed to find the NIXL wheel after building it.")

    # First, run auditwheel to bundle all other dependencies
    run_command(
        [
            sys.executable,
            "-m",
            "auditwheel",
            "repair",
            "--exclude",
            "libplugin_UCX.so",
            unrepaired_wheel,
            f"--wheel-dir={WHEELS_CACHE_HOME}",
        ],
        env=build_env,
    )

    repaired_wheel = find_nixl_wheel_in_cache(WHEELS_CACHE_HOME)
    if not repaired_wheel:
        raise RuntimeError("Failed to find repaired wheel from auditwheel.")

    # Now, unpack the repaired wheel to perform surgery on it
    wheel_unpack_dir = os.path.join(temp_wheel_dir, "wheel_unpack")
    if os.path.exists(wheel_unpack_dir):
        shutil.rmtree(wheel_unpack_dir)
    os.makedirs(wheel_unpack_dir)
    run_command(["unzip", "-q", repaired_wheel, "-d", wheel_unpack_dir])

    # Find the main NIXL extension file to inspect its dependencies
    nixl_extension_search = glob.glob(os.path.join(wheel_unpack_dir, "nixl", "*.so"))
    if not nixl_extension_search:
        raise RuntimeError("Could not find main NIXL .so extension file.")
    nixl_extension_file = nixl_extension_search[0]

    # Find the .libs directory
    libs_dir_search = glob.glob(os.path.join(wheel_unpack_dir, "*.libs"))
    if not libs_dir_search:
        raise RuntimeError("Could not find .libs directory in unpacked wheel.")
    libs_dir = libs_dir_search[0]

    # Find the incorrect libfabric that auditwheel bundled
    incorrect_lib_basename = None
    for lib in os.listdir(libs_dir):
        if "libfabric" in lib:
            incorrect_lib_basename = lib
            break

    # Only perform replacement if we found a library to replace
    if incorrect_lib_basename:
        incorrect_lib_path = os.path.join(libs_dir, incorrect_lib_basename)
        print(
            f"--> Found and deleting incorrect bundled library: {incorrect_lib_basename}",
            flush=True,
        )
        os.remove(incorrect_lib_path)

        # Find the correct, pre-built libfabric library
        lf_lib_path = os.path.join(lf_install_path, "lib")
        libfabric_so_files = glob.glob(os.path.join(lf_lib_path, "libfabric.so.1.*"))
        if not libfabric_so_files:
            raise RuntimeError(f"Could not find libfabric.so.1.* in {lf_lib_path}")
        correct_libfabric_src = max(libfabric_so_files, key=len)
        correct_libfabric_basename = os.path.basename(correct_libfabric_src)

        # Copy it into the wheel's .libs directory
        print(
            f"--> Copying correct library '{correct_libfabric_basename}' into wheel",
            flush=True,
        )
        shutil.copy2(correct_libfabric_src, os.path.join(libs_dir, incorrect_lib_path))

        # Use patchelf to update the dependency link in the main NIXL extension
        # print(f"--> Patching NIXL extension to link against '{correct_libfabric_basename}'", flush=True)
        # run_command(['patchelf', '--replace-needed', incorrect_lib_basename, correct_libfabric_basename, nixl_extension_file])
    else:
        print(
            "--> Warning: Did not find a bundled libfabric to remove. It might have been excluded.",
            flush=True,
        )

    # Repack the corrected wheel, overwriting the one from auditwheel
    print(
        f"--> Repacking corrected wheel to '{os.path.basename(repaired_wheel)}'",
        flush=True,
    )
    run_command(["zip", "-r", repaired_wheel, "."], cwd=wheel_unpack_dir)

    # --- Cleanup ---
    shutil.rmtree(temp_wheel_dir)

    # --- Final Installation ---
    newly_built_wheel = find_nixl_wheel_in_cache(WHEELS_CACHE_HOME)
    if not newly_built_wheel:
        raise RuntimeError("Failed to find the repaired NIXL wheel.")

    print(
        f"--> Successfully built self-contained wheel: {os.path.basename(newly_built_wheel)}. Now installing...",
        flush=True,
    )
    install_command = [sys.executable, "-m", "pip", "install", newly_built_wheel]
    if args.force_reinstall:
        install_command.insert(-1, "--force-reinstall")

    run_command(install_command)
    print("--- NIXL installation complete ---", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build and install UCX and NIXL dependencies."
    )
    parser.add_argument(
        "--force-reinstall",
        action="store_true",
        help="Force rebuild and reinstall of UCX and NIXL even if they are already installed.",
    )
    args = parser.parse_args()
    build_and_install_prerequisites(args)
