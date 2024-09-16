import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, 
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
import torch
from transformers import BitsAndBytesConfig
import os

class TextClassificationPipeline:
    def __init__(self, model_path, lora_r, lora_alpha, lora_dropout, use_rslora, init_lora_weights, bias, use_dora, load_in_8bit, load_in_4bit, llm_int8_threshold, bnb_4bit_quant_type, bnb_4bit_use_double_quant, mixed, autocast_adapter_dtype, max_len=200):
        self.model_path = model_path
        self.max_len = max_len
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_rslora = use_rslora
        self.bias = bias
        self.init_lora_weights = init_lora_weights
        self.use_dora = use_dora 
        self.load_in_8bit =  load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.llm_int8_threshold = llm_int8_threshold
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        self.mixed = mixed
        self.autocast_adapter_dtype = autocast_adapter_dtype
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
        
        self.hyperparameters = {
            'model_path': self.model_path,
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'use_rslora': self.use_rslora,
            'init_lora_weights': self.init_lora_weights,
            'bias': self.bias,
            'use_dora': self.use_dora,
            'load_in_8bit': self.load_in_8bit,
            'load_in_4bit': self.load_in_4bit,
            'llm_int8_threshold': self.llm_int8_threshold,
            'bnb_4bit_quant_type': self.bnb_4bit_quant_type,
            'bnb_4bit_use_double_quant': self.bn    b_4bit_use_double_quant,
            'mixed': self.mixed,
            'autocast_adapter_dtype': self.autocast_adapter_dtype
        }
        self.metrics = {
            "true_labels": [],
            "predicted_labels": [],
            "probabilities": [], 
            "attention_weights": [],  
        }

    def load_data(self, train_path=None, test_path=None, text_column=None, label_column=None, dataset_name=None):
        self.text_column = text_column  
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
        quantization_config = BitsAndBytesConfig(
                load_in_8bit = self.load_in_8bit, 
                load_in_4bit = self.load_in_4bit, 
                llm_int8_threshold = self.llm_int8_threshold, 
                bnb_4bit_quant_type= self.bnb_4bit_quant_type, 
                bnb_4bit_use_double_quant = self.bnb_4bit_use_double_quant,
            )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, 
            num_labels=len(id2label), 
            id2label=id2label, 
            label2id=label2id,
            quantization_config=quantization_config,
            device_map="auto"
            # Removed output_attentions=True
        )
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model = prepare_model_for_kbit_training(self.model)
        
        target_modules = self.model_target_modules.get(model_type, None)
        if not target_modules:
            raise ValueError(f"Target modules for model type '{model_type}' not found. Please check the configuration.")
        
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, 
            r=self.lora_r, 
            lora_alpha=self.lora_alpha, 
            lora_dropout=self.lora_dropout, 
            bias=self.bias,
            target_modules=target_modules, 
            use_rslora = self.use_rslora, 
            init_lora_weights = self.init_lora_weights, 
            use_dora = self.use_dora
        )
        
        self.model = get_peft_model(self.model, peft_config )
        
    def compute_metrics(self, p):
        logits = p.predictions
        labels = p.label_ids
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.argmax(logits, axis=1)
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        self.metrics["true_labels"].extend(labels.tolist())
        self.metrics["predicted_labels"].extend(preds.tolist())
        self.metrics["probabilities"].extend(probs.tolist())

        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='weighted', zero_division=0)
        recall = recall_score(labels, preds, average='weighted', zero_division=0)
        f1 = f1_score(labels, preds, average='weighted', zero_division=0)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(torch.tensor(logits), torch.tensor(labels)).item()

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "loss": loss}
        
    def train(self, output_dir="output_model", num_train_epochs=3, learning_rate=2e-5, batch_size=32, experiment_name="experiment", output_excel_file="metrics_output.xlsx"):
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
            push_to_hub=False,
            report_to=["none"],
            logging_dir='./logs',
            logging_steps=10
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        self.trainer.train()
        self.evaluate_and_save(self.trainer, self.tokenized_dataset["test"], output_excel_file, num_train_epochs, experiment_name)
        
    def evaluate_and_save(self, trainer, test_dataset_sample, output_excel_file, num_epochs, experiment_name):
        predictions = trainer.predict(test_dataset_sample, ignore_keys=["past_key_values", "hidden_states", "attentions"])
        logits = predictions.predictions
        labels = predictions.label_ids
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.argmax(logits, axis=1) 
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy() 

        # Extract attention weights
        batch_size_eval = trainer.args.per_device_eval_batch_size
        num_samples = len(test_dataset_sample)
        all_attention_weights = []

        for i in range(0, num_samples, batch_size_eval):
            batch = test_dataset_sample.select(range(i, min(i+batch_size_eval, num_samples)))
            inputs = self.tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=self.max_len)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                model_outputs = self.model(**inputs, output_attentions=True)
                attentions = model_outputs.attentions 
            batch_size_actual = attentions[0].size(0)  

            for j in range(batch_size_actual):
                attention_per_example = []
                for layer_attention in attentions:
                    attention_example = layer_attention[j] 
                    attention_example = attention_example.mean(dim=0) 
                    attention_per_example.append(attention_example.cpu().numpy().tolist())
                all_attention_weights.append(attention_per_example)

        
        self.metrics["attention_weights"].extend(all_attention_weights)
        metrics = self.compute_metrics(predictions)
        self.save_metrics_to_excel(metrics, experiment_name, output_excel_file, num_epochs)

    def save_metrics_to_excel(self, metrics, experiment_name, output_excel_file, num_epochs):
        min_length = min(len(self.metrics["true_labels"]), len(self.metrics["predicted_labels"]), len(self.metrics["probabilities"]), len(self.metrics["attention_weights"]))
        true_labels = self.metrics["true_labels"][:min_length]
        predicted_labels = self.metrics["predicted_labels"][:min_length]
        probabilities = self.metrics["probabilities"][:min_length]
        attention_weights = self.metrics["attention_weights"][:min_length]

        # Prepare hyperparameters and metrics for DataFrame
        data = {
            'Experiment Name': [experiment_name] * min_length,
            'Model Path': [self.model_path] * min_length,
            'LoRA Rank (r)': [self.lora_r] * min_length,
            'LoRA Alpha': [self.lora_alpha] * min_length,
            'LoRA Dropout': [self.lora_dropout] * min_length,
            'Use RsLoRA': [self.use_rslora] * min_length,
            'Init LoRA Weights': [self.init_lora_weights] * min_length,
            'Bias': [self.bias] * min_length,
            'Use DoRA': [self.use_dora] * min_length,
            'Load in 8-bit': [self.load_in_8bit] * min_length,
            'Load in 4-bit': [self.load_in_4bit] * min_length,
            'LLM Int8 Threshold': [self.llm_int8_threshold] * min_length,
            '4-bit Quant Type': [self.bnb_4bit_quant_type] * min_length,
            '4-bit Use Double Quant': [self.bnb_4bit_use_double_quant] * min_length,
            'Mixed Precision': [self.mixed] * min_length,
            'Autocast Adapter Dtype': [self.autocast_adapter_dtype] * min_length,
            'Accuracy': [metrics.get('accuracy')] * min_length,
            'Precision': [metrics.get('precision')] * min_length,
            'Recall': [metrics.get('recall')] * min_length,
            'F1 Score': [metrics.get('f1')] * min_length,
            'Loss': [metrics.get('loss')] * min_length,
            'True Labels': true_labels,
            'Predicted Labels': predicted_labels,
            'Probabilities': [str(p) for p in probabilities],  # Convert to string for Excel
            'Attention Weights': [str(a) for a in attention_weights],  # Convert to string for Excel
        }

        df = pd.DataFrame(data)
        
        if not os.path.exists(output_excel_file):
            with pd.ExcelWriter(output_excel_file, mode="w", engine='openpyxl') as writer:
                df.to_excel(writer, index=False, header=True)
        else:
            with pd.ExcelWriter(output_excel_file, mode="a", engine='openpyxl', if_sheet_exists="overlay") as writer:
                if 'Sheet1' in writer.book.sheetnames:
                    startrow = writer.sheets["Sheet1"].max_row
                else:
                    startrow = 0
                df.to_excel(writer, index=False, header=False, startrow=startrow)

        print(f"Metrics saved to {output_excel_file} for {experiment_name}")

if __name__ == "__main__":
    # All our datasets have the text_column and label_column as text and label
    # Adjust accordingly
    text_column = "text"
    label_column = "label"
    
    # Dataset paths used for our experiments, add your own datasets as needed
    # Remember to also include the label mapping for the labels in your dataset
    dataset_paths = {

        'causal': ('train_causal.csv', 'test_causal.csv'),
        'agnews': ('train_agnews.csv', 'test_agnews.csv'),
        'imdb': ('train_sample_Imdb_2000.csv', 'test_sample_imdb_2000.csv'),
        'dailog': ('daily_dialog_train.csv', 'dailydialog_test.csv'),
        'financialsentiment': ('train_financial_sentiment.csv', 'test_financial_sentiment.csv'),
        'emotion': ('train_emotion_sentiment.csv', 'test_emotion_sentiment.csv')
    }
    
    # Label mappings for each datasets
    label_mappings = {
        'causal': {
            'id2label': {0: "Non-Causal", 1: "Causal"},
            'label2id': {"Non-Causal": 0, "Causal":1}
        },
        'agnews': {
            'id2label': {0: "WORLD", 1: "SPORTS", 2: "BUSINESS", 3: "SCIENCE"},
            'label2id': {"WORLD": 0, "SPORTS": 1, "BUSINESS": 2, "SCIENCE": 3}
        },
        'imdb': {
            'id2label': {0: "NEGATIVE", 1: "POSITIVE"},
            'label2id': {"NEGATIVE": 0, "POSITIVE": 1}
        },
        'dailog': {
            'id2label': {0: "none", 1: "anger", 2: "disgust", 3: "fear", 4: "happiness", 5: "sadness", 6: "surprise"},
            'label2id': {"none":0, "anger":1, "disgust":2, "fear":3, "happiness":4, "sadness":5, "surprise":6}
        },
        'financialsentiment': {
            'id2label': {0: "Bearish", 1: "Bullish", 2: "Neutral"},
            'label2id': {"Bearish": 0, "Bullish": 1,"Neutral": 2}
        },
        'emotion': {
            'id2label': {0: "SADNESS", 1: "JOY", 2: "LOVE", 3: "ANGER", 4: "FEAR", 5: "SURPRISE"},
            'label2id': {"SADNESS":0, "JOY":1, "LOVE":2, "ANGER":3, "FEAR":4, "SURPRISE":5}
        }
    }

    # Define the possible values for each hyperparameter

    possible_values = {

        "model_paths": [
            "facebook/opt-125m",
            'mistralai/Mistral-7B-Instruct-v0.3',
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "facebook/opt-350m",
            # Add model options here  as need, you also need to add the corresponding target modules

        ],
        # Some bias options may not be applicable to some models
        "bias": ["none", "all", "lora_only"],
        "bias": ["none"],
        "init_lora_weights": ['olora', True, False],
        "lora_r": [2, 4, 6,8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
        "use_rslora": [True, False],
        "lora_dropout": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "bnb_4bit_use_double_quant": [True, False],
        "mixed": [True, False],
        "autocast_adapter_dtype": [False, True],
        "bnb_4bit_quant_type": ["fp4", "nf4"],
        "use_dora": [True, False],
        "llm_int8_threshold": [6.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 55.0, 60.0],
        "bit_precision_options": [
            {'load_in_8bit': True, 'load_in_4bit': False},
            {'load_in_8bit': False, 'load_in_4bit': True},
        ],
    }
    # Other hyperparameters can be specified here
    max_len = 200
    num_train_epochs = 3
    learning_rate = 2e-5
    batch_size = 32

    
    def run_experiments_for_hyperparameter(hyperparameter_name, hyperparameter_list):
        for dataset_name, (train_path, test_path) in dataset_paths.items():
            id2label = label_mappings[dataset_name]['id2label']
            label2id = label_mappings[dataset_name]['label2id']
            for model_path in possible_values['model_paths']:
                for bit_precision in possible_values['bit_precision_options']:
                    
                    if bit_precision['load_in_8bit'] == bit_precision['load_in_4bit']:
                        continue
                    # Set default hyperparameters
                    hyperparameters = {
                        'model_path': model_path,
                        'lora_r': possible_values['lora_r'][0],
                        'lora_alpha': possible_values['lora_r'][0],
                        'lora_dropout': possible_values['lora_dropout'][0],
                        'use_rslora': possible_values['use_rslora'][0],
                        'init_lora_weights': possible_values['init_lora_weights'][0],
                        'bias': possible_values['bias'][0],
                        'use_dora': possible_values['use_dora'][0],
                        'load_in_8bit': bit_precision['load_in_8bit'],
                        'load_in_4bit': bit_precision['load_in_4bit'],
                        'llm_int8_threshold': possible_values['llm_int8_threshold'][0],
                        'bnb_4bit_quant_type': possible_values['bnb_4bit_quant_type'][0],
                        'bnb_4bit_use_double_quant': possible_values['bnb_4bit_use_double_quant'][0],
                        'mixed': possible_values['mixed'][0],
                        'autocast_adapter_dtype': possible_values['autocast_adapter_dtype'][0]
                    }

                    # Update the hyperparameter being tested
                    for value in hyperparameter_list:
                        hyperparameters[hyperparameter_name] = value

                        
                        experiment_name = f"{dataset_name}_{model_path.replace('/', '_')}_bit_{'8' if hyperparameters['load_in_8bit'] else '4'}bit_{hyperparameter_name}_{value}"

                        output_excel_file = f"metrics_output_accrossdatasets_{experiment_name}.xlsx"

                        print(f"Running experiment: {experiment_name}")
                        pipeline = TextClassificationPipeline(
                            model_path=hyperparameters['model_path'],
                            lora_r=hyperparameters['lora_r'],
                            lora_alpha=hyperparameters['lora_alpha'],
                            lora_dropout=hyperparameters['lora_dropout'],
                            use_rslora=hyperparameters['use_rslora'],
                            init_lora_weights=hyperparameters['init_lora_weights'],
                            bias=hyperparameters['bias'],
                            use_dora=hyperparameters['use_dora'],
                            load_in_8bit=hyperparameters['load_in_8bit'],
                            load_in_4bit=hyperparameters['load_in_4bit'],
                            llm_int8_threshold=hyperparameters['llm_int8_threshold'],
                            bnb_4bit_quant_type=hyperparameters['bnb_4bit_quant_type'],
                            bnb_4bit_use_double_quant=hyperparameters['bnb_4bit_use_double_quant'],
                            mixed=hyperparameters['mixed'],
                            autocast_adapter_dtype=hyperparameters['autocast_adapter_dtype'],
                            max_len=max_len
                        )

                        pipeline.load_data(
                            train_path=train_path, 
                            test_path=test_path, 
                            text_column=text_column, 
                            label_column=label_column
                        )

                        pipeline.preprocess_data()
                        pipeline.prepare_model(id2label=id2label, label2id=label2id)

                        # Train and save metrics
                        pipeline.train(
                            output_dir=f"output_model_{experiment_name}",
                            num_train_epochs=num_train_epochs,
                            learning_rate=learning_rate,
                            batch_size=batch_size,
                            experiment_name=experiment_name,
                            output_excel_file=output_excel_file
                        )

    # Now, run experiments for each hyperparameter separately
    hyperparameters_to_test = [
        'bias',
        'init_lora_weights',
        'lora_r',
        'use_rslora',
        'lora_dropout',
        'bnb_4bit_use_double_quant',
        'bnb_4bit_quant_type',
        'use_dora',
        'llm_int8_threshold'
    ]



    for hyperparameter_name in hyperparameters_to_test:
        print(f"Testing hyperparameter: {hyperparameter_name}")
        run_experiments_for_hyperparameter(hyperparameter_name, possible_values[hyperparameter_name])
