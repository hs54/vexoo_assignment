import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

def run_finetuning():
    model_id = "meta-llama/Llama-3.2-1B"
    
    # 1. Load GSM8K Dataset
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main")
    train_subset = dataset['train'].select(range(3000))

    # 2. Tokenization Setup
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        inputs = [f"Question: {q}\nAnswer: " for q in examples['question']]
        model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['answer'], padding="max_length", truncation=True, max_length=512)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_data = train_subset.map(tokenize_function, batched=True)

    # 3. LoRA Configuration (Efficiency)
    peft_config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05, 
        task_type="CAUSAL_LM"
    )

    # 4. Model Initialization
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    model = get_peft_model(model, peft_config)

    # 5. Training Arguments
    args = TrainingArguments(
        output_dir="./llama-gsm8k-results",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=False, # Use bf16 if on A100/H100
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_data
    )

    print("Setup complete. Starting training loop simulation...")
    # trainer.train() # Final execution

if __name__ == "__main__":
    run_finetuning()
