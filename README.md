<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Install NIXL + UCX (gaudi) + libfabric 

- launch Gaudi docker

``` bash
docker run -it --rm --runtime=habana --name=1.22.0-vllm-nixl-ucx -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host --net=host --shm-size 10g  --privileged vault.habana.ai/gaudi-docker/1.22.0/ubuntu24.04/habanalabs/pytorch-installer-2.7.1:latest
```

- install nixl

``` bash
git clone https://github.com/intel-staging/nixl.git -b v0.6.0_UCX
cp -r nixl /tmp/nixl_source
cd nixl
python install_nixl.py
```

- test nixl

```bash
pip install pandas msgspec pyzmq
UCX_TLS=gaudi_gdr,rc,ud,ib python nixl_api_test.py --nixl_backend UCX --block-size 128 --device-type hpu 2>&1 | tee nixl_UCX.log
```

- test with vllm

(Optional) Install VLLM + VLLM-gaudi

```bash
cd ..
git clone https://github.com/vllm-project/vllm-gaudi
cd vllm-gaudi
export VLLM_COMMIT_HASH=$(git show "origin/vllm/last-good-commit-for-vllm-gaudi:VLLM_STABLE_COMMIT" 2>/dev/null)

# Build vLLM from source for empty platform, reusing existing torch installation
git clone https://github.com/vllm-project/vllm
cd vllm
git checkout $VLLM_COMMIT_HASH
pip install -r <(sed '/^[torch]/d' requirements/build.txt)
VLLM_TARGET_DEVICE=empty pip install --no-build-isolation -e .
cd ..

cd vllm-gaudi
pip install -e .
cd ..

pip install lm_eval pytest pytest_asyncio
```

Run accuracy test

```bash
cd vllm-gaudi/tests/unit_tests/
NIXL_BUFFER_DEVICE=hpu VLLM_NIXL_BACKEND=UCX bash run_accuracy_test.sh
```

