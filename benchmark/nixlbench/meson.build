# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

project('nixlbench', 'CPP', version: '0.6.0',
    default_options: ['buildtype=release',
                'werror=true',
                'cpp_std=c++17',
                'prefix=/usr/local/nixlbench'],
    meson_version: '>= 0.64.0'
)

# set up some global vars for compiler, platform, configuration, etc.
cpp = meson.get_compiler('cpp')
fs = import('fs')

# Allow overriding paths through environment variables
# CUDA
cuda_inc_path = get_option('cudapath_inc')
cuda_lib_path = get_option('cudapath_lib')
cuda_stub_path = get_option('cudapath_stub')
# ETCD
etcd_inc_path = get_option('etcd_inc_path')
etcd_lib_path = get_option('etcd_lib_path')
# NIXL
nixl_path = get_option('nixl_path')
# NVSHMEM
nvshmem_inc_path = get_option('nvshmem_inc_path')
nvshmem_lib_path = get_option('nvshmem_lib_path')

# Find required libraries
# NIXL
host_cpu_family = host_machine.cpu_family()
host_system = host_machine.system().to_lower()

if host_system != 'linux' or host_cpu_family not in ['x86_64', 'aarch64']
    error('This build only supports Linux on x86_64 or aarch64 architectures.')
endif

nixl_lib_path = join_paths(nixl_path, get_option('libdir'))
nixl_lib = cpp.find_library('nixl', dirs: [nixl_lib_path])
nixl_build = cpp.find_library('nixl_build', dirs: [nixl_lib_path])
nixl_serdes = cpp.find_library('serdes', dirs: [nixl_lib_path])

if not nixl_lib.found() or not nixl_build.found() or not nixl_serdes.found()
    error('NIXL Libraries not found. Exiting.')
endif

# CUDA
cuda_available = false
if cuda_lib_path == ''
    cuda_dep = dependency('cuda', required : false, modules : [ 'cudart', 'cuda' ])
    if not cuda_dep.found()
        # Meson is not detecting CUDA reliably on ARM, fallback to default
        cuda_home = run_command('bash', '-c', 'echo $CUDA_HOME', check: true).stdout().strip()
        if cuda_home == ''
            cuda_home = '/usr/local/cuda'
        endif
        cuda_lib = cuda_home + '/lib64'
        cuda_inc = cuda_home + '/include'
        cuda_stub = cuda_lib + '/stubs'
        if fs.exists(cuda_lib) and fs.exists(cuda_inc)
            cuda_dep = declare_dependency(
              link_args : ['-L' + cuda_lib, '-L' + cuda_stub, '-lcuda', '-lcudart'],
              include_directories : include_directories(cuda_inc))
            if cuda_dep.found()
                message('Found CUDA installation through fallback method:', cuda_home)
            endif
        endif
    endif
    if cuda_dep.found()
        cuda_available = true
    else
        warning('CUDA not found. VRAM support will be disabled.')
    endif
else
    message('cuda lib path ', cuda_lib_path)
    if cuda_stub_path == ''
        cuda_stub_path = cuda_lib_path + '/stubs'
    endif
    cuda_dep = declare_dependency(
        link_args : ['-L' + cuda_lib_path, '-L' + cuda_stub_path, '-lcuda', '-lcudart'],
        include_directories : include_directories(cuda_inc_path))
    cuda_available = true
endif

cuda_fabric_available = false
if cuda_available
    if cpp.has_header_symbol('cuda.h', 'CU_MEM_HANDLE_TYPE_FABRIC',
                             args: cuda_inc_path != '' ? '-I' + cuda_inc_path : [])
        cuda_fabric_available = true
    endif
endif

# SynapseAI (Habana Gaudi) dependency detection
synapse_inc_path = get_option('synapsepath_inc')
synapse_lib_path = get_option('synapsepath_lib')

if synapse_lib_path == ''
    #use default path
    # Try to find both libSynapse and hl-thunk libraries
    synapse_lib = cpp.find_library('Synapse',
                                 dirs: ['/usr/lib/habanalabs', '/usr/local/lib/habanalabs'],
                               required: false)
    hlthunk_lib = cpp.find_library('hl-thunk',
                               dirs: ['/usr/lib/habanalabs', '/usr/local/lib/habanalabs'],
                               required: false)
else
    synapse_lib = cpp.find_library('Synapse',
                                 dirs: [synapse_lib_path],
                               required: false)
    hlthunk_lib = cpp.find_library('hl-thunk',
                               dirs: [synapse_lib_path],
                               required: false)
endif

if synapse_inc_path == ''
  #use default path
  synapse_inc_path = '/usr/include/habanalabs/'
endif

# SynapseAI support requires both libraries
synapseai_dep = dependency('', required: false)  # Initialize as not found
if synapse_lib.found() and hlthunk_lib.found()
    synapseai_dep = declare_dependency(dependencies: [synapse_lib, hlthunk_lib])
elif hlthunk_lib.found()
    # Fallback to just hl-thunk if libSynapse not available
    synapseai_dep = hlthunk_lib
endif

if synapseai_dep.found()
    # Create proper dependency with include paths (including DRM path for habanalabs headers)
    synapseai_dep = declare_dependency(
        dependencies: synapseai_dep,
        include_directories: [
            include_directories('/usr/include/drm'),
            include_directories(synapse_inc_path)
        ]
    )
    message('Found SynapseAI support for Habana Gaudi devices')
    synapseai_available = true
else
    synapseai_available = false
    warning('SynapseAI not found. Habana Gaudi device support will be disabled.')
endif

# GFlags
gflags_dep = dependency('gflags', required: true)

# OpenMP
openmp_dep = dependency('openmp', required: true)
# Try pkg-config first
etcd_dep = dependency('etcd-cpp-api', required : false)
if not etcd_dep.found()
    # Fallback: manual configuration
    #   message('etcd-cpp-api not found via pkg-config, using manual configuration')

    # Check if we have the library files
    etcd_lib = meson.get_compiler('cpp').find_library('etcd-cpp-api',
                    dirs: ['/usr/local/lib'],
                                  required: false)

      if etcd_lib.found()
            etcd_dep = declare_dependency(
                    include_directories: include_directories('/usr/local/include'),
                          dependencies: [etcd_lib],
            # Add any required dependencies for etcd-cpp-api
            link_args: []  # Add any additional link args if needed
            )
            message('etcd-cpp-api found manually in /usr/local/lib')
       else
            etcd_dep = disabler()
            message('etcd-cpp-api not found anywhere')
      endif
endif
etcd_available = etcd_dep.found()
if etcd_available
    add_project_arguments('-DHAVE_ETCD', language: 'cpp')
else
    message('ETCD C++ client library not found. Disabling ETCD runtime.')
endif

etcd_inc = etcd_inc_path
nixl_inc = nixl_path + '/include'
nvshmem_inc = nvshmem_inc_path

inc_dir = include_directories('.', './src/', nixl_inc)

nvshmem_available = false
if nvshmem_lib_path != ''
    if cpp.has_header('nvshmem.h', include_directories: include_directories(nvshmem_inc))
        nvshmem_lib = cpp.find_library('nvshmem', dirs: [nvshmem_lib_path])
        nvshmem_host_lib = cpp.find_library('nvshmem_host', dirs: [nvshmem_lib_path])
        if nvshmem_lib.found()
            nvshmem_available = true
            inc_dir = include_directories('.', './src/', nixl_inc, nvshmem_inc)
            add_project_arguments('-DHAVE_NVSHMEM', language: 'cpp')
            add_project_arguments('-Wno-unused-variable', language: 'cpp')
        endif
    endif
endif

if cuda_available
    add_project_arguments('-DHAVE_CUDA', language: 'cpp')
endif

if cuda_fabric_available
    add_project_arguments('-DHAVE_CUDA_FABRIC', language: 'cpp')
endif

if synapseai_available
    add_project_arguments('-DHAVE_SYNAPSEAI', language: 'cpp')
endif

# Subprojects
if synapseai_available
   subdir('src/synapseai')
endif
subdir('src/utils')
subdir('src/runtime')
subdir('src/worker')

# Configure header file
configure_file(
    output: 'config.h',
    configuration: {
        'HAVE_ETCD': etcd_available ? '1' : '0',
        'HAVE_NVSHMEM': nvshmem_available ? '1' : '0',
        'HAVE_CUDA': cuda_available ? '1' : '0',
        'HAVE_CUDA_FABRIC': cuda_fabric_available ? '1' : '0',
        'HAVE_SYNAPSEAI': synapseai_available ? '1' : '0',
    },
    install: true,
    install_dir: get_option('includedir') / 'nixlbench'
)

deps = [gflags_dep, nixl_lib, nixl_build, nixl_serdes, openmp_dep]
args = []
if etcd_available
    deps += [etcd_dep]
endif
if cuda_available
    deps += [cuda_dep]
endif

if synapseai_available
    deps += [synapseai_dep]
    message('add synapseai_dep')
endif
if nvshmem_available
    deps += [nvshmem_lib]
    args += [
        '-Xcompiler', '-fopenmp',
        '-Xlinker', '-rpath=/usr/local/cuda/lib64',
        '-Xlinker', '--allow-shlib-undefined',
        '-lcudart',
        '-lcudadevrt',
    ]
endif

#if not etcd_available
#    error('No runtime available or not found')
#endif

if nvshmem_available
    # Use nvcc directly for compilation and linking
    nvcc = find_program('nvcc')
    nvcc_args = []
    if etcd_available
        if etcd_inc != ''
            nvcc_args += ['-I' + etcd_inc]
        endif
    endif
    nvcc_args += ['-I' + nixl_inc]
    nvcc_args += ['-I' + nvshmem_inc]
    nvcc_args += ['-I.', '-I./src/', '-I../src/']
    nvcc_args += ['-I' + meson.current_build_dir()]
    if etcd_available
        if etcd_lib_path != ''
            nvcc_args += ['-L' + etcd_lib_path]
        endif
    endif
    nvcc_args += ['-L' + nixl_lib_path]
    nvcc_args += ['-L' + nvshmem_lib_path]
    nvcc_args += ['-L' + meson.current_build_dir() + '/src/utils']
    nvcc_args += ['-L' + meson.current_build_dir() + '/src/runtime']
    nvcc_args += ['-L' + meson.current_build_dir() + '/src/worker']
    nvcc_args += ['-lnixl', '-lnixl_build', '-lserdes', '-lgflags', '-lcuda', '-lcudart', '-lnvshmem']
    nvcc_args += args
    nvcc_cmd_files = [
                 meson.current_build_dir() + '/src/utils/libutils.a.p/utils.cpp.o',
                 meson.current_build_dir() + '/src/runtime/libruntime.a.p/runtime.cpp.o',
                 meson.current_build_dir() + '/src/worker/libworker.a.p/worker.cpp.o',
                 meson.current_build_dir() + '/src/worker/nixl/libnixl_worker.a.p/nixl_worker.cpp.o',
                 meson.current_build_dir() + '/src/worker/nvshmem/libnvshmemWorker.a.p/nvshmem_worker.cpp.o'
                 ]

    if etcd_available
        nvcc_args += ['-letcd-cpp-api-core', '-lcpprest']
        etcd_rt = meson.current_build_dir() + '/src/runtime/etcd/libetcd_rt.a.p/etcd_rt.cpp.o'
        nvcc_cmd_files += etcd_rt
    endif

    nvcc_command = [nvcc, nvcc_args, '-o', '@OUTPUT@', '@INPUT@']
    nvcc_command += nvcc_cmd_files

    custom_target('nixlbench',
        input: 'src/main.cpp',
        output: 'nixlbench',
        command: nvcc_command,
        build_by_default: true,
        install: true,
        install_dir: get_option('bindir'),
        depends: [nixlbench_runtimes, utils_lib, worker_libs])
else
   if synapseai_available
       executable('nixlbench', 'src/main.cpp',
                   include_directories: inc_dir,
                   link_with: [nixlbench_runtimes, utils_lib, worker_libs, synapseaiutils_lib],
                   dependencies: deps,
                   link_args: args,
                   install: true,
                   install_dir: get_option('bindir'))
     else
       executable('nixlbench', 'src/main.cpp',
                   include_directories: inc_dir,
                   link_with: [nixlbench_runtimes, utils_lib, worker_libs],
                   dependencies: deps,
                   link_args: args,
                   install: true,
                   install_dir: get_option('bindir'))
     endif
endif
