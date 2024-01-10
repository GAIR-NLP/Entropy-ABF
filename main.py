import torch
from transformers import LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import patch.EABF as EABF


def load_eabf_model(model_path: str, use_flash_attention=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_flash_attention:
        from patch import replace_llama_attn_with_flash_attn

        replace_llama_attn_with_flash_attn()  # use flash attention
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        rope_scaling={"type": "eabf", "factor": 4},
    ).to(device)
    EABF.apply_eabf(model)
    return model, tokenizer


def test(model, tokenizer):
    from prompt import prompt

    input = tokenizer(prompt, return_tensors="pt")
    prompt_length = input.input_ids.shape[-1]

    output = model.generate(
        input_ids=input.input_ids.to(model.device),
        max_new_tokens=200,
        use_cache=False,
    )[0][prompt_length:]
    output = tokenizer.batch_decode([output], skip_special_tokens=True)[0]

    print("Model Output: ", output)
    print("---------------------------------------\n")
    print(f"Prompt Length = {prompt_length}")


if __name__ == "__main__":
    model_path = "/cpfs01/shared/GAIR/GAIR_hdd/ykzhang/ours/abf_longchat16k10k"
    model, tokenizer = load_eabf_model(model_path)

    test(model, tokenizer)
