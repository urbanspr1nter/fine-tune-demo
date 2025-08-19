from unsloth import FastModel
import torch

checkpoint_path = "/home/rngo/code/llm_demo/gemma-3-1b-finetuned-cp/checkpoint-6400"

model, tokenizer = FastModel.from_pretrained(
    checkpoint_path,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    load_in_8bit=False,
    device_map="cpu"
)

model.save_pretrained_merged(
    "gemma-3-1b-finetuned-merged",
    tokenizer=tokenizer
)