# Extending LLMs' Context Window with 100 Samples

This is the official repo for "Extending LLMs' Context Window with 100 Samples". [Preprint](http://arxiv.org/abs/2401.07004)

## Introduction

We introduce 'Entropy-Aware ABF' that supports efficient context window extension of RoPE-based LLMs with only 100 samples. The repository contains code and data to replicate our results.

## Model and Data

We release long-context [Llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) extended with our method trained with different data amounts on 🤗Hugging Face:

| Data | Link |
| ---: | :--- |
| 0.1k | [🤗eabf-llama2-7b-chat-0.1k](https://huggingface.co/Arist12/eabf-llama2-7b-chat-0.1k)  |
|   1k | [🤗eabf-llama2-7b-chat-1k](https://huggingface.co/Arist12/eabf-llama2-7b-chat-1k)    |
| 3.5k | [🤗eabf-llama2-7b-chat-3.5k](https://huggingface.co/Arist12/eabf-llama2-7b-chat-3.5k)    |

We also release our training data on 🤗[Hugging Face Datasets](https://huggingface.co/datasets/Arist12/EABF-ShareGPT-Long-3.5k).

## Quick Guide

### Use Entropy-Aware ABF

To use our code, your [transformers](https://github.com/huggingface/transformers) library should be version 4.31 or higher.

We replicate the paper summarization test proposed in NTK-Aware scaling's [blog](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/) to serve as a sanity check.

In short, to load the LLaMA model with our method, you should first import the required packages:

```python
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import patch.eabf as eabf
```

Then, you can load the model by using the right `rope_scaling` argument and our monkey patching function:

```python
model = LlamaForCausalLM.from_pretrained(MODEL_NAME_OR_PATH, ..., rope_scaling={"type": "eabf", "factor": 4})
eabf.apply_eabf(model)
```

### Replicate Observation of Attention Scores

Other RoPE-based LLMs might or might not follow the same attention scores pattern as [Llama-2-7b-chat](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), we release our code for retrieving attention scores and computing the 'attention entropy' so that users can apply our method tailored to their model.
