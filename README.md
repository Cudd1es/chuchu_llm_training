# chuchu_llm_training
Use unsloth/gemma-2-9b to train chatbot that plays as CHU²

ref: https://docs.unsloth.ai

to install requirements:

```
conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate unsloth_env

pip install unsloth
```
