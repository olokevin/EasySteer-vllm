It's recommended to prepare a new environment for this vLLM implementation

Please follow the[documentation](https://docs.astral.sh/uv/#getting-startedhttps:/) to install uv.

## Environment

Install the customized RKV vLLM implementation

```
# Create a new conda environment
conda create -n easysteer python=3.10 -y
conda activate easysteer

# Install with pre-compiled version (recommended)
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/cede942b87b5d8baa0b95447f3e87e3c600ff5f5/vllm-0.9.2rc2.dev34%2Bgcede942b8-cp38-abi3-manylinux1_x86_64.whl
pip install --editable .
pip install transformers==4.53.1

```

build the dependencies of the evaluation toolkit separately:

```
cd evaluation/latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt
```

## Single prompt testing scripts

R1KV / R2KV_SLOW only: `python test_rkv.py`

Steering (SEAL): `python test_seal.py`

R1KV / R2KV_SLOW + SEAL: `python test_rkv_seal.py`
