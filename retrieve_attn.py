import torch
from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import json
import os
from tqdm import tqdm
import math


def compute_attention_entropy():
    layers = [0, 1, 2, 9, 16, 24, 31]
    entropy_lst = [[] for _ in range(len(layers))]

    with open(f"output/attn_{position}.json", "r") as f:
        data = json.load(f)
        attn_scores = [data[f"layer-{i}"] for i in layers]
        for i in range(len(layers)):
            entropy = 0
            for val in attn_scores[i]:
                entropy -= val * math.log(val, math.e)
            entropy_lst[i].append(entropy)

    print(entropy_lst)


def collect_probability(attn_scores):
    last_token_attn = attn_scores[:, -1].softmax(
        dim=1
    )  # shape: (head_size, seq_length)

    # select the attention score on the zero-th head
    attn = last_token_attn[0].to(torch.float32)  # shape: (seq_length)
    return attn.detach().cpu().tolist()


if __name__ == "__main__":
    position = 4096  # set the truncation length

    # load model
    model_path = "path/to/model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
    )
    model = model.eval()

    with open("data/data_128.jsonl", "r") as f:
        inputs = [json.loads(line)["text"] for line in f]

    documents_attn = []
    with torch.no_grad():
        for input in tqdm(inputs):
            input = tokenizer(
                input, truncation=True, max_length=position, return_tensors="pt"
            )
            input = input.to(device)

            output = model(**input, output_attentions=True)
            attentions = output.attentions

            layer_attn = []
            selected_layers = [x for x in range(32)]

            for layer in selected_layers:
                # retrieve the attention scores for the 4,096th token
                attn_scores = attentions[
                    layer
                ]  # shape: (1, head_size, seq_length, seq_length)
                attn_scores.squeeze_()  # shape: (head_size, seq_length, seq_length)
                attn = collect_probability(attn_scores)

                layer_attn.append(attn)
            documents_attn.append(layer_attn)

    documents_attn_pt = torch.tensor(
        documents_attn
    )  # shape: (document_count, selected_head_num, seq_len)
    avg_attn_lst = documents_attn_pt.mean(
        dim=0
    ).tolist()  # shape: (selected_head_num, seq_len)

    out_folder = "output/"
    os.makedirs(out_folder, exist_ok=True)

    obj = {}
    with open(f"{out_folder}/attn_{position}.json", "w") as f:
        for i, attn in enumerate(avg_attn_lst):
            obj[f"layer-{selected_layers[i]}"] = attn
        json.dump(obj, f)
