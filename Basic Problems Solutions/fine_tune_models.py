from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

def fine_tune_model(dataset, label, model_name="gpt2", output_dir="output"):
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add a padding token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Filter the dataset by label
    data = dataset.filter(lambda example: example['label'] == label)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length')

    tokenized_data = data.map(tokenize_function, batched=True)

    # Use DataCollatorWithPadding to handle varying sequence lengths
    data_collator = DataCollatorWithPadding(tokenizer)

    # Prepare LoRA configuration
    lora_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=["mlp.c_fc", "mlp.c_proj"],  # Adjusted modules
        lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{label}", 
        per_device_train_batch_size=4, 
        num_train_epochs=3, 
        save_steps=10_000, 
        save_total_limit=2,
    )

    # Trainer
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=tokenized_data['train'],
        data_collator=data_collator  # Added data collator here
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(f"{output_dir}/{label}")
    tokenizer.save_pretrained(f"{output_dir}/{label}")

if __name__ == "__main__":
    dataset = load_dataset("ag_news")
    labels = {
        1: "World"
        #2: "Sports",
        #3: "Business",
        #4: "SciTech"
    }
    
    for label, name in labels.items():
        print(f"Fine-tuning model for {name}...")
        fine_tune_model(dataset, label)
