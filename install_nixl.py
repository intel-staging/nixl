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
UCX_DIR = os.path.join('/tmp', 'ucx_source')
LIBFABRIC_DIR = os.path.join('/tmp', 'libfabric_source')
NIXL_DIR = os.path.join('/tmp', 'nixl_source')
UCX_INSTALL_DIR = os.path.join('/tmp', 'ucx_install')
LIBFABRIC_INSTALL_DIR = os.path.join('/tmp', 'libfabric_install')
NIXL_INSTALL_DIR = os.path.join('/tmp', 'nixl_install')

# --- Repository and Version Configuration ---
UCX_REPO_URL = 'https://github.com/openucx/ucx.git'
UCX_BRANCH = 'v1.19.x'
LIBFABRIC_REPO_URL = 'https://github.com/ofiwg/libfabric.git'
LIBFABRIC_REF = 'v1.21.0'  # Using a recent stable tag
NIXL_REPO_URL = 'https://github.com/intel-staging/nixl.git'
NIXL_BRANCH = 'v0.6.0_OFI'


# --- Helper Functions ---
def run_command(command, cwd='.', env=None):
    """Helper function to run a shell command and check for errors."""
    print(f"--> Running command: {' '.join(command)} in '{cwd}'", flush=True)
    subprocess.check_call(command, cwd=cwd, env=env)


def is_pip_package_installed(package_name):
    """Checks if a package is installed via pip without raising an exception."""
    result = subprocess.run([sys.executable, '-m', 'pip', 'show', package_name],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
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
        print("WARNING: Not running as root. Skipping system dependency installation.", flush=True)
        print("Please ensure the following packages are installed on your system:", flush=True)
        print("  patchelf build-essential git cmake ninja-build autotools-dev automake meson libtool libtool-bin",
              flush=True)
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
        "zip"
    ]
    run_command(['apt-get', 'update'])
    run_command(['apt-get', 'install', '-y'] + apt_packages)
    print("--- System dependencies installed successfully. ---\n", flush=True)

def install_nixl():
    # Save original directory
    original_cwd = os.getcwd()
    # Set environment variables
    os.environ["LIBUCX_ROOT"] = UCX_INSTALL_DIR
    os.environ["LIBNIXL_ROOT"] = NIXL_INSTALL_DIR
    os.environ["LIBFABRIC_ROOT"] = LIBFABRIC_INSTALL_DIR

    os.environ["PKG_CONFIG_PATH"] = (
        f"{os.environ['LIBFABRIC_ROOT']}/lib/pkgconfig:"
        f"{os.environ['LIBUCX_ROOT']}/lib/pkgconfig:"
        f"{os.environ['LIBNIXL_ROOT']}/lib/pkgconfig:"
        + os.environ.get("PKG_CONFIG_PATH", "")
    )

    os.environ["CPLUS_INCLUDE_PATH"] = (
        f"{os.environ['LIBNIXL_ROOT']}/include:" +
        os.environ.get("CPLUS_INCLUDE_PATH", "")
    )

    os.environ["C_INCLUDE_PATH"] = (
        f"{os.environ['LIBNIXL_ROOT']}/include:" +
        os.environ.get("C_INCLUDE_PATH", "")
    )

    os.environ["LDFLAGS"] = f"-L{os.environ['LIBNIXL_ROOT']}/lib " + os.environ.get("LDFLAGS", "")
    os.environ["LD_LIBRARY_PATH"] = (
        f"{os.environ['LIBNIXL_ROOT']}/lib:" +
        os.environ.get("LD_LIBRARY_PATH", "")
    )

    try:

        # Change directory
        os.chdir(NIXL_DIR)

        # Run pip installs
        subprocess.run(["pip", "install", "--upgrade", "meson", "pybind11", "patchelf"], check=True)
        subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)

        # Meson setup
        subprocess.run([
            "meson", "setup",
            "--wipe",
            f"--prefix={os.environ['LIBNIXL_ROOT']}",
            "--buildtype=release",
            "-Ddisable_gds_backend=true",
            f"-Dlibfabric_path={os.environ['LIBFABRIC_ROOT']}",
            f"-Ducx_path={os.environ['LIBUCX_ROOT']}",
            "builddir", "."
        ], check=True)

        # Build and install
        os.chdir("builddir")
        subprocess.run(["ninja"], check=True)
        subprocess.run(["ninja", "install"], check=True)
        subprocess.run(["ldconfig"], check=True)
        os.chdir("..")
        # Install python package
        subprocess.run(["pip", "install", "."], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}: {e.cmd}")
        raise

    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

    finally:
        # Return to original directory
        os.chdir(original_cwd)

def build_and_install_prerequisites(args):
    """Builds UCX and NIXL from source, creating a self-contained wheel."""

    # ... (initial checks and setup are unchanged) ...
    if not args.force_reinstall and is_pip_package_installed('nixl'):
        print("--> NIXL is already installed. Nothing to do.", flush=True)
        return

    cached_wheel = find_nixl_wheel_in_cache(WHEELS_CACHE_HOME)
    if not args.force_reinstall and cached_wheel:
        print(f"\n--> Found self-contained wheel: {os.path.basename(cached_wheel)}.", flush=True)
        print("--> Installing from cache, skipping all source builds.", flush=True)
        install_command = [sys.executable, '-m', 'pip', 'install', cached_wheel]
        run_command(install_command)
        print("\n--- Installation from cache complete. ---", flush=True)
        return

    print("\n--> No installed package or cached wheel found. Starting full build process...", flush=True)
    install_system_dependencies()
    ucx_install_path = os.path.abspath(UCX_INSTALL_DIR)
    print(f"--> Using wheel cache directory: {WHEELS_CACHE_HOME}", flush=True)
    os.makedirs(WHEELS_CACHE_HOME, exist_ok=True)

    # -- Step 1: Build UCX from source --
    print("\n[1/3] Configuring and building UCX from source...", flush=True)
    if not os.path.exists(UCX_DIR):
        run_command(['git', 'clone', UCX_REPO_URL, UCX_DIR])
    ucx_source_path = os.path.abspath(UCX_DIR)
    run_command(['git', 'checkout', 'v1.19.x'], cwd=ucx_source_path)
    run_command(['./autogen.sh'], cwd=ucx_source_path)
    configure_command = [
        './configure',
        f'--prefix={ucx_install_path}',
        '--enable-shared',
        '--disable-static',
        '--disable-doxygen-doc',
        '--enable-optimizations',
        '--enable-cma',
        '--enable-devel-headers',
        '--with-verbs',
        '--enable-mt',
    ]
    run_command(configure_command, cwd=ucx_source_path)
    run_command(['make', '-j', str(os.cpu_count() or 1)], cwd=ucx_source_path)
    run_command(['make', 'install'], cwd=ucx_source_path)
    print("--- UCX build and install complete ---", flush=True)

    # -- Step 2: Build Libfabric from source --
    print(f"\n[2/3] Configuring and building Libfabric (ref: {LIBFABRIC_REF}) from source...", flush=True)
    if not os.path.exists(LIBFABRIC_DIR):
        run_command(['git', 'clone', LIBFABRIC_REPO_URL, LIBFABRIC_DIR])
    run_command(['git', 'checkout', LIBFABRIC_REF], cwd=LIBFABRIC_DIR)
    run_command(['./autogen.sh'], cwd=LIBFABRIC_DIR)
    configure_command_lf = [
        './configure',
        f'--prefix={LIBFABRIC_INSTALL_DIR}',
        '--enable-verbs', '--enable-shm', '--enable-sockets', '--enable-tcp',
        '--with-synapseai=/usr/include/habanalabs' # As requested
    ]
    run_command(configure_command_lf, cwd=LIBFABRIC_DIR)
    run_command(['make', '-j', str(os.cpu_count() or 1)], cwd=LIBFABRIC_DIR)
    run_command(['make', 'install'], cwd=LIBFABRIC_DIR)
    print("--- Libfabric build and install complete ---", flush=True)
    
    
    # -- Step 3: Build NIXL wheel from source --
    print(f"\n[3/3] Building NIXL (branch: {NIXL_BRANCH}) wheel from source...", flush=True)
    if not os.path.exists(NIXL_DIR):
        run_command(['git', 'clone', '--branch', NIXL_BRANCH, NIXL_REPO_URL, NIXL_DIR])
    install_nixl()
    print("--- NIXL installation complete ---", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and install UCX and NIXL dependencies.")
    parser.add_argument('--force-reinstall',
                        action='store_true',
                        help='Force rebuild and reinstall of UCX and NIXL even if they are already installed.')
    args = parser.parse_args()
    build_and_install_prerequisites(args)
