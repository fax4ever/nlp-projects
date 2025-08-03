import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset, load_dataset
import os
from typing import Dict, List
import json
import re


class LLMSentenceSplitter:
    """
    Fine-tune an LLM using QLoRA for Italian sentence splitting.
    """
    
    def __init__(self, model_name: str = "microsoft/Phi-3.5-mini-instruct"):
        """
        Initialize the LLM sentence splitter.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
    def setup_model_and_tokenizer(self):
        """Setup tokenizer and model with QLoRA configuration."""
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model with quantization for QLoRA
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True,  # 4-bit quantization
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # LoRA rank
            lora_alpha=32,  # LoRA scaling parameter
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Adjust based on model
        )
        
        # Apply LoRA to model
        self.peft_model = get_peft_model(self.model, lora_config)
        self.peft_model.print_trainable_parameters()
        
    def convert_token_data_to_instruction_format(self, dataset_name: str = "fax4ever/manzoni") -> Dataset:
        """
        Convert token-level dataset to instruction format for LLM training.
        
        Args:
            dataset_name: HuggingFace dataset name
            
        Returns:
            Dataset formatted for instruction tuning
        """
        
        # Load your existing dataset
        dataset = load_dataset(dataset_name)
        
        def create_instruction_examples(examples):
            """Convert token sequences to instruction format."""
            instructions = []
            
            for tokens, labels in zip(examples['tokens'], examples['labels']):
                # Reconstruct text from tokens
                text = " ".join(tokens)
                
                # Create expected output with sentence boundaries
                sentences = []
                current_sentence = []
                
                for token, label in zip(tokens, labels):
                    current_sentence.append(token)
                    if label == 1:  # End of sentence
                        sentences.append(" ".join(current_sentence).strip())
                        current_sentence = []
                
                # Add remaining tokens if any
                if current_sentence:
                    sentences.append(" ".join(current_sentence).strip())
                
                # Create instruction format
                instruction = {
                    "instruction": "Dividi il seguente testo italiano in frasi. Rispondi con una frase per riga.",
                    "input": text,
                    "output": "\n".join([f"{i+1}. {sentence}" for i, sentence in enumerate(sentences)])
                }
                
                instructions.append(instruction)
                
            return instructions
        
        # Convert to instruction format
        train_instructions = create_instruction_examples(dataset['train'])
        val_instructions = create_instruction_examples(dataset['validation'])
        
        # Create new dataset
        train_dataset = Dataset.from_list(train_instructions)
        val_dataset = Dataset.from_list(val_instructions)
        
        return {"train": train_dataset, "validation": val_dataset}
    
    def format_training_examples(self, examples):
        """Format examples for causal language modeling."""
        
        def create_prompt(instruction, input_text, output_text):
            return f"""<|system|>
Sei un esperto di linguistica italiana specializzato nella segmentazione delle frasi.

<|user|>
{instruction}

Testo: {input_text}

<|assistant|>
{output_text}<|end|>"""
        
        prompts = []
        for instruction, input_text, output in zip(
            examples['instruction'], 
            examples['input'], 
            examples['output']
        ):
            prompt = create_prompt(instruction, input_text, output)
            prompts.append(prompt)
        
        # Tokenize
        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    def train(self, output_dir: str = "./llm-sentence-splitter", num_epochs: int = 3):
        """
        Fine-tune the LLM for sentence splitting.
        
        Args:
            output_dir: Directory to save the fine-tuned model
            num_epochs: Number of training epochs
        """
        
        # Setup model and tokenizer
        print("Setting up model and tokenizer...")
        self.setup_model_and_tokenizer()
        
        # Prepare dataset
        print("Preparing dataset...")
        datasets = self.convert_token_data_to_instruction_format()
        
        # Tokenize datasets
        print("Tokenizing datasets...")
        train_dataset = datasets['train'].map(
            self.format_training_examples,
            batched=True,
            remove_columns=datasets['train'].column_names
        )
        
        val_dataset = datasets['validation'].map(
            self.format_training_examples,
            batched=True,
            remove_columns=datasets['validation'].column_names
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Not masked language modeling
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,  # Small batch size for memory efficiency
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,  # Effective batch size = 8
            learning_rate=2e-4,  # Higher LR for LoRA
            num_train_epochs=num_epochs,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_steps=200,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            warmup_steps=100,
            fp16=True,  # Mixed precision training
            dataloader_pin_memory=False,
            push_to_hub=True,
            hub_token=os.getenv("HF_TOKEN"),
            report_to=None,  # Disable wandb/tensorboard for simplicity
        )
        
        # Trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save the model
        print(f"Saving model to {output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Push to hub
        if os.getenv("HF_TOKEN"):
            trainer.push_to_hub(commit_message="Fine-tuned for Italian sentence splitting")
    
    def inference(self, text: str, model_path: str = None) -> List[str]:
        """
        Perform inference on a text to split it into sentences.
        
        Args:
            text: Input text to split
            model_path: Path to fine-tuned model (if None, uses base model)
            
        Returns:
            List of sentences
        """
        
        if model_path and not self.peft_model:
            # Load fine-tuned model
            self.setup_model_and_tokenizer()
            # Load LoRA weights
            from peft import PeftModel
            self.peft_model = PeftModel.from_pretrained(self.model, model_path)
        
        elif not self.peft_model:
            # Use base model
            self.setup_model_and_tokenizer()
        
        # Create prompt
        prompt = f"""<|system|>
Sei un esperto di linguistica italiana specializzato nella segmentazione delle frasi.

<|user|>
Dividi il seguente testo italiano in frasi. Rispondi con una frase per riga.

Testo: {text}

<|assistant|>
"""
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract sentences from response
        response_part = response.replace(prompt, "").strip()
        
        # Parse numbered sentences
        sentences = []
        for line in response_part.split('\n'):
            line = line.strip()
            if re.match(r'\d+\.\s+', line):
                sentence = re.sub(r'^\d+\.\s+', '', line)
                sentences.append(sentence)
        
        return sentences
    
    def evaluate_on_sample(self, num_samples: int = 10):
        """
        Evaluate the model on a few samples from the validation set.
        
        Args:
            num_samples: Number of samples to evaluate
        """
        
        # Load validation data
        dataset = load_dataset("fax4ever/manzoni")['validation']
        
        for i in range(min(num_samples, len(dataset))):
            example = dataset[i]
            
            # Reconstruct original text
            original_text = " ".join(example['tokens'])
            
            # Get model predictions
            predicted_sentences = self.inference(original_text)
            
            # Get ground truth sentences
            ground_truth_sentences = []
            current_sentence = []
            
            for token, label in zip(example['tokens'], example['labels']):
                current_sentence.append(token)
                if label == 1:
                    ground_truth_sentences.append(" ".join(current_sentence).strip())
                    current_sentence = []
            
            if current_sentence:
                ground_truth_sentences.append(" ".join(current_sentence).strip())
            
            print(f"\n--- Example {i+1} ---")
            print(f"Original: {original_text}")
            print(f"Ground Truth ({len(ground_truth_sentences)} sentences):")
            for j, sent in enumerate(ground_truth_sentences):
                print(f"  {j+1}. {sent}")
            print(f"Predicted ({len(predicted_sentences)} sentences):")
            for j, sent in enumerate(predicted_sentences):
                print(f"  {j+1}. {sent}")


def main():
    """Example usage"""
    
    # Initialize splitter
    splitter = LLMSentenceSplitter("microsoft/Phi-3.5-mini-instruct")
    
    # Train the model
    splitter.train(
        output_dir="./italian-sentence-splitter-llm",
        num_epochs=2
    )
    
    # Evaluate on samples
    splitter.evaluate_on_sample(num_samples=5)


if __name__ == "__main__":
    main()