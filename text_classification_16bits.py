# Import necessary libraries
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import load_dataset

class TextClassificationPipeline:
    def __init__(self, model_name, text_column, label_column, num_labels, id2label, label2id, learning_rate, batch_size, num_epochs, dataset_name=None, train_file=None, test_file=None):
        self.dataset_name = dataset_name
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
        self.num_epochs = num_epochs

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # Load dataset and model
        self.dataset = self.load_dataset()
        self.model = self.load_model()

    def load_dataset(self):
        if self.dataset_name:
            # Load the dataset from Hugging Face
            return load_dataset(self.dataset_name)
        elif self.train_file and self.test_file:
            # Load the dataset from local CSV files
            train_data = pd.read_csv(self.train_file)
            test_data = pd.read_csv(self.test_file)
            return DatasetDict({
                "train": Dataset.from_pandas(train_data),
                "test": Dataset.from_pandas(test_data)
            })
        else:
            raise ValueError("Either dataset_name or both train_file and test_file must be provided.")

    def preprocess_function(self, examples):
        # Tokenize the input examples
        return self.tokenizer(examples[self.text_column], truncation=True, padding=True)

    def tokenize_dataset(self):
        # Apply preprocessing to the dataset
        return self.dataset.map(self.preprocess_function, batched=True)

    def load_model(self):
        # Load the specified model
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels, id2label=self.id2label, label2id=self.label2id
        )

    def compute_metrics(self, p):
        # Compute the evaluation metrics
        preds, labels = p
        preds = np.argmax(preds, axis=1)
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, average='weighted')
        recall = recall_score(labels, preds, average='weighted')
        f1 = f1_score(labels, preds, average='weighted')
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def train(self, output_dir="model_output"):
        # Tokenize dataset
        tokenized_dataset = self.tokenize_dataset()

        # Set training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
        )

        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        # Train the model
        trainer.train()

        # Optionally, save the model
        # self.model.push_to_hub("your_model_repository_name")

# Example usage
if __name__ == "__main__":
    # Specify the model details
    model_name = "distilbert-base-uncased"
    text_column = "text"
    label_column = "label"
    num_labels = 2
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    learning_rate = 2e-5
    batch_size = 16
    num_epochs = 1
    dataset_name = "imdb"

    # Example with Hugging Face dataset
    pipeline_hf = TextClassificationPipeline(
        model_name=model_name,
        text_column=text_column,
        label_column=label_column,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        dataset_name=dataset_name
    )

    # Train the model on Hugging Face dataset
    pipeline_hf.train()

    ## Uncomment for using local files
    
    # # Example with local CSV files
    # train_file = "replace with your csv train path"
    # test_file = "replace with your csv test path"

#     pipeline_local = TextClassificationPipeline(
#         model_name=model_name,
#         text_column=text_column,
#         label_column=label_column,
#         num_labels=num_labels,
#         id2label=id2label,
#         label2id=label2id,
#         learning_rate=learning_rate,
#         batch_size=batch_size,
#         num_epochs=num_epochs,
#         train_file=train_file,
#         test_file=test_file
#     )

#     # Train the model on local dataset
#     pipeline_local.train()
