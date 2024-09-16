import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TextClassificationPipeline:
    def __init__(self, model_name, text_column, label_column, num_labels, id2label, label2id, learning_rate, batch_size, train_file, test_file, weight_decay, adam_epsilon, fp16, fp16_opt_level, max_grad_norm):
        self.train_file = train_file
        self.test_file = test_file
        self.model_name = model_name
        self.text_column = text_column
        self.label_column = label_column
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = label2id
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.fp16 = fp16
        self.fp16_opt_level = fp16_opt_level
        self.max_grad_norm = max_grad_norm
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        self.hyperparameters = {
            'model_name': self.model_name,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'weight_decay': self.weight_decay,
            'adam_epsilon': self.adam_epsilon,
            'fp16': self.fp16,
            'fp16_opt_level': self.fp16_opt_level,
            'max_grad_norm': self.max_grad_norm,
        }
        self.dataset = self.load_dataset()
        self.model = self.load_model()
        self.metrics = {
            "true_labels": [],
            "predicted_labels": [],
            "probabilities": [],  
            "attention_weights": [],  
        }
        self.summary_metrics = []  

    def load_dataset(self):
        train_data = pd.read_csv(self.train_file)
        test_data = pd.read_csv(self.test_file)

        if self.text_column not in train_data.columns or self.label_column not in train_data.columns:
            raise ValueError(f"Text column '{self.text_column}' or label column '{self.label_column}' not found in CSV file.")
        if self.text_column not in test_data.columns or self.label_column not in test_data.columns:
            raise ValueError(f"Text column '{self.text_column}' or label column '{self.label_column}' not found in CSV file.")

        return DatasetDict({
            "train": Dataset.from_pandas(train_data),
            "test": Dataset.from_pandas(test_data)
        })

    def preprocess_function(self, examples):
        return self.tokenizer(examples[self.text_column], truncation=True, padding=True, max_length=200)

    def tokenize_dataset(self):
        return self.dataset.map(self.preprocess_function, batched=True)

    def load_model(self):
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels, id2label=self.id2label, label2id=self.label2id
        )

    def compute_metrics(self, p):
        logits, labels = p  
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
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "loss": loss
        }

    def save_metrics_to_excel(self, output_file, experiment_name="", num_epochs=1):
        min_length = min(len(self.metrics["true_labels"]), len(self.metrics["predicted_labels"]), len(self.metrics["probabilities"]), len(self.metrics["attention_weights"]))
        true_labels = self.metrics["true_labels"][:min_length]
        predicted_labels = self.metrics["predicted_labels"][:min_length]
        probabilities = self.metrics["probabilities"][:min_length]
        attention_weights = self.metrics["attention_weights"][:min_length]
        hyperparameters_df = {k: [v] * len(true_labels) for k, v in self.hyperparameters.items()}

        
        metrics_df = pd.DataFrame({
            "Dataset": [experiment_name.split('_')[0]] * len(true_labels),
            "Experiment Name": [experiment_name] * len(true_labels),
            "Epochs": [num_epochs] * len(true_labels),
            "True Labels": true_labels,
            "Predicted Labels": predicted_labels,
            "Probabilities": [str(p) for p in probabilities],  
            "Attention Weights": [str(a) for a in attention_weights]
        })
        for key, value in hyperparameters_df.items():
            metrics_df[key] = value
        if not os.path.exists(output_file):
            with pd.ExcelWriter(output_file, mode="w", engine='openpyxl') as writer:
                metrics_df.to_excel(writer, index=False, header=True)
        else:
            with pd.ExcelWriter(output_file, mode="a", engine='openpyxl', if_sheet_exists="overlay") as writer:
                if 'Sheet1' in writer.book.sheetnames:
                    startrow = writer.sheets["Sheet1"].max_row
                else:
                    startrow = 0
                metrics_df.to_excel(writer, index=False, header=False, startrow=startrow)

        print(f"Metrics saved to {output_file} for {experiment_name}")

    def evaluate_and_save(self, trainer, test_dataset_sample, output_excel_file, num_epochs, experiment_name):
        predictions = trainer.predict(test_dataset_sample)
        logits, labels = predictions.predictions, predictions.label_ids
        preds = np.argmax(logits, axis=1)  
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()  
        inputs = self.tokenizer([example[self.text_column] for example in test_dataset_sample], return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}  

        with torch.no_grad():
            model_outputs = self.model(**inputs, output_attentions=True)
            attentions = model_outputs.attentions  

        for idx, (logit, label, pred, prob) in enumerate(zip(logits, labels, preds, probs)):
            batch_idx = idx % attentions[0].size(1) 
            attention_per_example = [torch.mean(att[:, batch_idx, :, :], dim=(0, 1)).cpu().numpy().tolist() for att in attentions]
            self.metrics["true_labels"].append(int(label))
            self.metrics["predicted_labels"].append(int(pred))
            self.metrics["probabilities"].append(prob.tolist())  
            self.metrics["attention_weights"].append(attention_per_example)  

        metrics = self.compute_metrics((predictions.predictions, predictions.label_ids))
        summary_metric = {
            "Experiment Name": experiment_name,
            "Epochs": num_epochs,
            "Dataset": experiment_name.split('_')[0],
            "Accuracy": metrics["accuracy"],
            "F1 Score": metrics["f1"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "Loss": metrics["loss"]
        }
        summary_metric.update(self.hyperparameters)
        self.summary_metrics.append(summary_metric)
        self.save_metrics_to_excel(output_excel_file, experiment_name=experiment_name, num_epochs=num_epochs)

    def train(self, output_dir="model_output", output_excel_file="metrics_output.xlsx", train_sample_size=100, test_sample_size=100, num_epochs=1, experiment_name=""):
        self.metrics = {
            "true_labels": [],
            "predicted_labels": [],
            "probabilities": [],
            "attention_weights": [],  
        }

    
        tokenized_dataset = self.tokenize_dataset()
    
        # We have de-activated the sampling, if want to include sampling, add your code here
        train_dataset_sample = tokenized_dataset["train"].shuffle(seed=42)
        test_dataset_sample = tokenized_dataset["test"].shuffle(seed=42)

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            fp16=self.fp16,
            fp16_opt_level=self.fp16_opt_level,
            adam_epsilon=self.adam_epsilon,
            weight_decay=self.weight_decay,
            max_grad_norm=self.max_grad_norm,
            num_train_epochs=num_epochs,  
            evaluation_strategy="epoch",
            save_strategy="epoch",  
            load_best_model_at_end=True,  
            logging_dir='./logs',
            logging_steps=10,
            push_to_hub=False,
            report_to=["none"],
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset_sample,  
            eval_dataset=test_dataset_sample,   
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()
        self.evaluate_and_save(trainer, test_dataset_sample, output_excel_file, num_epochs, experiment_name)
        return self.metrics

    def save_summary_to_excel(self, output_file):
        summary_df = pd.DataFrame(self.summary_metrics)
        with pd.ExcelWriter(output_file, mode="a", engine='openpyxl', if_sheet_exists="replace") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

        print(f"Summary saved to {output_file} in 'Summary' sheet")


# Example usage
if __name__ == "__main__":
    # These were the text and label columns in our dataset, adjust accordingly in your dataset
    text_column = "text"
    label_column = "label"

    # Define the possible values for each hyperparameter
    possible_values = {
        "model_names": [
            "distilbert-base-uncased",
            "distilbert/distilroberta-base",
            "facebook/opt-125m",
            "facebook/opt-350m", 
            # Add more models as needed
        ],
       "batch_size": [
            64, 60, 16, 20, 24, 28, 32, 36,40, 44, 48, 52, 56
        ],
        "learning_rate": [
            5e-5, 3e-5, 2e-5, 1e-5, 5e-6, 3e-6, 1e-6, 1e-4, 3e-4, 5e-4
        ],
 
        "num_train_epochs": [
            3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
        ],
        "weight_decay": [
           0.0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0
        ],
        "fp16": [
            True, False
        ],
        "fp16_opt_level": [
             "O0", "O1", "O2", "O3"
        ],
        "max_grad_norm": [
            0.0, 2.0, 4.0, 6.0, 8.0, 10.0
        ]
    }

    # Replace these with your paths and datasets
    dataset_paths = {
        'causal': ('train_causal.csv', 'test_causal.csv'),
        'agnews': ('train_agnews.csv', 'test_agnews.csv'),
        'imdb': ('train_sample_Imdb_2000.csv', 'test_sample_imdb_2000.csv'),
        'dailog': ('daily_dialog_train.csv', 'dailydialog_test.csv'),
        'financialsentiment': ('train_financial_sentiment.csv', 'test_financial_sentiment.csv'),
        'emotion': ('train_emotion_sentiment.csv', 'test_emotion_sentiment.csv')
    }
    

    # Replace these with the label mapping in your dataset
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

    def run_experiments_for_hyperparameter(hyperparameter_name, hyperparameter_list):
        for dataset_name, (train_file, test_file) in dataset_paths.items():
            id2label = label_mappings[dataset_name]['id2label']
            label2id = label_mappings[dataset_name]['label2id']
            num_labels = len(id2label)

            for model_name in possible_values['model_names']:
                output_file = f"metrics_output_accross_datasets_{hyperparameter_name}_{dataset_name}_{model_name.replace('/', '_')}.xlsx"
                for value in hyperparameter_list:
                    # Set default hyperparameters
                    hyperparameters = {
                        'learning_rate': possible_values['learning_rate'][0],
                        'batch_size': possible_values['batch_size'][0],
                        'num_train_epochs': possible_values['num_train_epochs'][0],
                        'weight_decay': possible_values['weight_decay'][0],
                        'adam_epsilon': 1e-8,
                        'fp16': possible_values['fp16'][0],
                        'fp16_opt_level': possible_values['fp16_opt_level'][0],
                        'max_grad_norm': possible_values['max_grad_norm'][0],
                    }

                    hyperparameters[hyperparameter_name] = value

                    experiment_name = f"{dataset_name}_experiment_{hyperparameter_name}_{value}_{model_name.replace('/', '_')}"
                    print(f"Running experiment: {experiment_name}")
                    pipeline = TextClassificationPipeline(
                        model_name=model_name,
                        text_column=text_column,
                        label_column=label_column,
                        num_labels=num_labels,
                        id2label=id2label,
                        label2id=label2id,
                        learning_rate=hyperparameters['learning_rate'],
                        batch_size=hyperparameters['batch_size'],
                        train_file=train_file,
                        test_file=test_file,
                        weight_decay=hyperparameters['weight_decay'],
                        adam_epsilon=hyperparameters['adam_epsilon'],
                        fp16=hyperparameters['fp16'],
                        fp16_opt_level=hyperparameters['fp16_opt_level'],
                        max_grad_norm=hyperparameters['max_grad_norm']
                    )

                    pipeline.train(
                        output_excel_file=output_file,
                        train_sample_size=100,
                        test_sample_size=100,
                        num_epochs=hyperparameters['num_train_epochs'],
                        experiment_name=experiment_name
                    )
                    pipeline.save_summary_to_excel(output_file=output_file)
                    
 
    #Run experiments for learning rate
    run_experiments_for_hyperparameter('learning_rate', possible_values['learning_rate'])

    #Run experiments for number of epochs
    run_experiments_for_hyperparameter('num_train_epochs', possible_values['num_train_epochs'])

    #Run experiments for weight decay
    run_experiments_for_hyperparameter('weight_decay', possible_values['weight_decay'])

    # Run experiments for fp16
    run_experiments_for_hyperparameter('fp16', possible_values['fp16'])

    # Run experiments for fp16_opt_level
    run_experiments_for_hyperparameter('fp16_opt_level', possible_values['fp16_opt_level'])

    # Run experiments for max_grad_norm
    run_experiments_for_hyperparameter('max_grad_norm', possible_values['max_grad_norm'])

    # Run for batch size

    run_experiments_for_hyperparameter('batch_size', possible_values['batch_size'])
