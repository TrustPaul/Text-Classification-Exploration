import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, 
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
import torch

class TextClassificationPipeline:
    def __init__(self, model_path, max_len=200, bit_precision="8bit"):
        self.model_path = model_path
        self.max_len = max_len
        self.bit_precision = bit_precision
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Mapping of model names to their respective target modules
        self.model_target_modules = {
            "opt": ["q_proj", "v_proj"],
            "falcon": ["query_key_value"],
            "phi": ["Wqkv", "fc1", "fc2"],
            "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3", "lm_head"],
            "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
            "zeply": ["q_proj", "k_proj", "v_proj", "o_proj"],
            # Add other models and their target modules here
        }
        
    def load_data(self, train_path=None, test_path=None, text_column=None, label_column=None, dataset_name=None):
        if dataset_name:
            dataset = load_dataset(dataset_name)
            
            def rename_columns(dataset, text_column, label_column):
                if text_column != "text":
                    dataset = dataset.rename_column(text_column, "text")
                if label_column != "label":
                    dataset = dataset.rename_column(label_column, "label")
                return dataset

            if "train" in dataset and "test" in dataset:
                self.dataset = DatasetDict({
                    "train": rename_columns(dataset["train"], text_column, label_column),
                    "test": rename_columns(dataset["test"], text_column, label_column)
                })
            else:
                train_test_split = dataset["train"].train_test_split(test_size=0.5)
                self.dataset = DatasetDict({
                    "train": rename_columns(train_test_split['train'], text_column, label_column),
                    "test": rename_columns(train_test_split['test'], text_column, label_column)
                })
        else:
            df_train = pd.read_csv(train_path)
            df_test = pd.read_csv(test_path)
            
            train_dataset = Dataset.from_pandas(df_train[[text_column, label_column]].rename(columns={text_column: "text", label_column: "label"}))
            test_dataset = Dataset.from_pandas(df_test[[text_column, label_column]].rename(columns={text_column: "text", label_column: "label"}))
            
            self.dataset = DatasetDict({
                "train": train_dataset,
                "test": test_dataset
            })
        
    def preprocess_data(self):
        def preprocess_function(examples):
            return self.tokenizer(examples["text"], truncation=True, max_length=self.max_len)
        
        self.tokenized_dataset = self.dataset.map(preprocess_function, batched=True)
        
    def prepare_model(self, id2label, label2id):
        load_in_8bit = self.bit_precision == "8bit"
        load_in_4bit = self.bit_precision == "4bit"

        # Infer model type from model_path using more robust substring checking
        model_type = None
        if "llama" in self.model_path.lower():
            model_type = "llama"
        elif "opt" in self.model_path.lower():
            model_type = "opt"
        elif "falcon" in self.model_path.lower():
            model_type = "falcon"
        elif "phi" in self.model_path.lower():
            model_type = "phi"
        elif "mistral" in self.model_path.lower():
            model_type = "mistral"
        elif "zeply" in self.model_path.lower():
            model_type = "zeply"
        else:
            raise ValueError(f"Model type for '{self.model_path}' not recognized. Please check the configuration.")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, num_labels=len(id2label), id2label=id2label, label2id=label2id,
            load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit
        )
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Retrieve target modules for the inferred model type
        target_modules = self.model_target_modules.get(model_type, None)
        if not target_modules:
            raise ValueError(f"Target modules for model type '{model_type}' not found. Please check the configuration.")
        
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=2, lora_alpha=16, lora_dropout=0.1, bias="none",
            target_modules=target_modules
        )
        
        self.model = get_peft_model(self.model, peft_config)
        
    def compute_metrics(self, p):
        preds, labels = p
        preds = np.argmax(preds, axis=1)
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='macro')
        recall = recall_score(labels, preds, average='macro')
        f1 = f1_score(labels, preds, average='macro')
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
        
    def train(self, output_dir="output_model", num_train_epochs=3, learning_rate=2e-5, batch_size=32):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )
        
        self.trainer.train()
        self.trainer.evaluate()

if __name__ == "__main__":
    # Some supported Models
    #meta-llama/Meta-Llama-3-8B-Instruct
    #mistralai/Mistral-7B-Instruct-v0.3
    #mistralai/Mistral-7B-Instruct-v0.2
    #meta-llama/Llama-2-7b-hf
    #meta-llama/Llama-2-13b-chat-hf
    #meta-llama/Llama-2-7b-chat-hf
    #HuggingFaceH4/zephyr-7b-beta
    #mistralai/Mistral-7B-v0.1
    #facebook/opt-125m
    # Set model path and options
    model_path = "meta-llama/Meta-Llama-3-8B-Instruct"  # Change as needed
    bit_precision = "4bit"  # or "8bit"
    max_len = 200  # Change as needed
    num_train_epochs = 1  # Change as needed

    # Option to load custom data or Hugging Face dataset
    use_custom_data = False  # Set to False to use Hugging Face dataset

    if use_custom_data:
        train_path = '/your_path/train.csv'
        test_path = '/your_path/test.csv'
        text_column = 'text'  # specify the text column name
        label_column = 'label'  # specify the label column name
        dataset_name = None
    else:
        train_path = None
        test_path = None
        dataset_name = "imdb"  # Example dataset name from Hugging Face
        text_column = 'text'  # specify the text column name for the Hugging Face dataset
        label_column = 'label'  # specify the label column name for the Hugging Face dataset

    # Choose labels for your specific task
    id2label = {0: "negative", 1: "positive"}
    label2id = {v: k for k, v in id2label.items()}
    
    # Initialize and run the pipeline
    pipeline = TextClassificationPipeline(model_path, max_len=max_len, bit_precision=bit_precision)
    pipeline.load_data(train_path=train_path, test_path=test_path, text_column=text_column, label_column=label_column, dataset_name=dataset_name)
    pipeline.preprocess_data()
    pipeline.prepare_model(id2label, label2id)
    pipeline.train(output_dir="my_awesome_model", num_train_epochs=num_train_epochs)
