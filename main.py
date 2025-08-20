from unsloth import FastModel
import torch
from datasets import load_dataset, Dataset
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig

dataset_name = "HuggingFaceTB/everyday-conversations-llama3.1-2k"
train_split_name = "train_sft"

def convert_to_conversations(example):
    return {"conversations": example["messages"]}

model, tokenizer = FastModel.from_pretrained(
    "unsloth/gemma-3-1b-pt",
    dtype=torch.bfloat16,
    max_seq_length=8192,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False
)

model = FastModel.get_peft_model(
    model,
    finetune_attention_modules=True,
    finetune_language_layers=True,
    finetune_mlp_modules=True,
    r=32,
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    modules_to_save=["lm_head", "embed_tokens"]
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-3"
)

dataset = load_dataset(dataset_name, split=train_split_name)
converted_dataset = [convert_to_conversations(example) for example in dataset] 

# converted_dataset = converted_dataset[:1]

def formatting_prompts_func(examples):
    conversations = examples["conversations"]

    texts = [
        tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        ).removeprefix('<bos>') for conversation in conversations
    ]

    return {"text": texts}

dataset = Dataset.from_list(converted_dataset)
dataset = dataset.map(
    formatting_prompts_func,
    batched=True
)

# print(dataset[:1])

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=None,
    args=SFTConfig(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        dataset_text_field="text",
        warmup_ratio=0.05,
        num_train_epochs=6,
        learning_rate=2e-4,
        optim="adamw_8bit",
        logging_steps=1,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        save_strategy="steps",
        save_steps=200,
        output_dir="gemma-3-1b-finetuned-cp"
    )
)

should_train = False

if should_train:
    trainer.train()

    model.save_pretrained_merged(
        "gemma-3-1b-finetuned-merged",
        tokenizer=tokenizer
    )
