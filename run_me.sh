# install the latest vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm

export VLLM_COMMIT=72d9c316d3f6ede485146fe5aabd4e61dbc59069
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/${VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
pip install --editable .