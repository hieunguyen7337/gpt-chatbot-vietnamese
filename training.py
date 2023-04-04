import os
import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model 
from datasets import Dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("./gptj6B_VI_model")
model = AutoModelForCausalLM.from_pretrained("./gptj6B_VI_model", device_map="auto", load_in_8bit=True)

# Add padding token if missing and resize token embeddings accordingly
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Freeze the model parameters and cast small parameters to fp32 for stability
for param in model.parameters():
    param.requires_grad = False
    if param.ndim == 1:
        param.data = param.data.to(torch.float32)

# Enable gradient checkpointing to reduce memory usage and enable input gradients
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# Cast output to float32
class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)

model.lm_head = CastOutputToFloat(model.lm_head)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# Print the number of trainable parameters in the model
print_trainable_parameters(model)

# Define PEFT config and update the model with it
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)

# Print the number of trainable parameters in the updated model
print_trainable_parameters(model)

# Define the training dataset
train_data = Dataset.from_dict({"text": ["Chào bạn, tên mình là", "Bạn tên là gì?"]})

# Define the trainer object
trainer = transformers.Trainer(
    model=model, 
    train_dataset=train_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        warmup_steps=10, 
        max_steps=20, 
        learning_rate=2e-4, 
        fp16=True,
        logging_steps=1, 
        output_dir='outputs'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# Train the model
trainer.train()

# Generate text using the trained model
prompt = "Tiềm năng của trí tuệ nhân tạo"
generated_text = model.generate(tokenizer(prompt, return_tensors="pt"), max_new_tokens=50)[0]

# Decode the generated text and print it
decoded_text = tokenizer.decode(generated_text, skip_special_tokens=True)
print(f"Generated text: {decoded_text}")
