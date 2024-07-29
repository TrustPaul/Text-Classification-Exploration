import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class TextCategorizer:
    MODEL_PROMPT_FORMATS = {
        "meta-llama/Meta-Llama-3-8B-Instruct": "system_user_assistant",
        "meta-llama/Meta-Llama-3-70B-Instruct": "system_user_assistant",
        "mistralai/Mixtral-8x7B-Instruct-v0.1": "inst_sys",
        "mistralai/Mistral-7B-Instruct-v0.2": "inst_sys",
        "mistralai/Mistral-7B-Instruct-v0.3": "inst_sys",
        "meta-llama/Llama-2-70b-hf": "inst_sys",
        "HuggingFaceH4/zephyr-7b-beta": "system_user",
        "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": "system_user",
        "meta-llama/Meta-Llama-3.1-8B-Instruct": "system_user_assistant",
        "meta-llama/Meta-Llama-3.1-70B-Instruct": "system_user_assistant"
    }

    def __init__(self, model_name: str, precision: str = "16-bit", temperature: float = 0.1, 
                 do_sample: bool = True, repetition_penalty: float = 1.1, max_new_tokens: int = 100):
        self.model_name = model_name
        self.precision = precision
        self.temperature = temperature
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self.pipeline = self._setup_pipeline()

    def _load_model_and_tokenizer(self):
        if self.precision == "4-bit":
            model = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_4bit=True)
        elif self.precision == "8-bit":
            model = AutoModelForCausalLM.from_pretrained(self.model_name, load_in_8bit=True)
        else:  # 16-bit is the default
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return model, tokenizer

    def _setup_pipeline(self):
        return pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="text-generation",
            temperature=self.temperature,
            do_sample=self.do_sample,
            repetition_penalty=self.repetition_penalty,
            return_full_text=True,
            max_new_tokens=self.max_new_tokens,
        )

    def _generate_prompt(self, text: str, user_prompt: str):
        system_prompt = "You are a language model trained by OpenAI that answers user questions."
        user_msg_1 = f"""
        {user_prompt}
        Here is the text to classify:
        Text: {text} 
        
        Your Response:
        """

        prompt_format = self.MODEL_PROMPT_FORMATS.get(self.model_name, "default")

        if prompt_format == "system_user_assistant":
            prompt = f"""
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                { system_prompt }<|eot_id|><|start_header_id|>user<|end_header_id|>

                { user_msg_1 }<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        elif prompt_format == "inst_sys":
            prompt = f"""<s>[INST] <<SYS>>
            {system_prompt}
            <</SYS>>
            {user_msg_1} [/INST]
            """
        elif prompt_format == "system_user":
            prompt = f"""<|im_start|>system
                                 {system_prompt}<|im_end|>
                                <|im_start|>user
                                {user_msg_1} <|im_end|>"""
        else:  # Default fallback
            prompt = f"{system_prompt}\n{user_msg_1}"
        return prompt

    def categorize_texts(self, texts, user_prompt: str):
        answers_generated = []
        for text in texts:
            prompt = self._generate_prompt(text, user_prompt)
            try:
                sequences = self.pipeline(prompt, max_new_tokens=self.max_new_tokens, do_sample=self.do_sample, top_k=5, return_full_text=False)
                answers = sequences[0]['generated_text']
            except Exception as e:
                answers = "Error in generating label"
            answers_generated.append(answers)
        return answers_generated

    def process_data(self, csv_path: str, user_prompt: str, num_samples: int = 10, text_column: str = 'text', label_column: str = None):
        data = pd.read_csv(csv_path)
        data = data.sample(n=num_samples)
        texts = data[text_column] if text_column in data.columns else data.iloc[:, 0]
        labels = data[label_column] if label_column and label_column in data.columns else None

        answers_generated = self.categorize_texts(texts, user_prompt)

        result = pd.DataFrame({text_column: texts, 'pred_label': answers_generated})
        if labels is not None:
            result[label_column] = labels
        return result

# Usage
if __name__ == "__main__":
    #mistralai/Mistral-7B-Instruct-v0.3
    #meta-llama/Meta-Llama-3-8B-Instruct
    #meta-llama/Meta-Llama-3-70B-Instruct
    #mistralai/Mixtral-8x7B-Instruct-v0.1
    #meta-llama/Llama-2-70b-hf
    #HuggingFaceH4/zephyr-7b-beta
    #NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO
    #meta-llama/Meta-Llama-3.1-8B-Instruct
    #meta-llama/Meta-Llama-3.1-70B-Instruct
    model_name = 'mistralai/Mistral-7B-Instruct-v0.3'  # Replace with the desired model name
    precision = '4-bit'  # Options: '4-bit', '8-bit', '16-bit' (default)
    csv_path = "/your_path/train.csv"
    num_samples = 10
    text_column = 'text'
    label_column = 'label'

    user_prompt = """Categorize the provided text into one of the specified categories:
    - World
    - Sports
    - Business
    - Science

    The category must be one of the following ['World', 'Sports', 'Business', 'Science']
    Return output as JSON format 'category': category """

    # Hyperparameters for the pipeline
    temperature = 0.1
    do_sample = True
    repetition_penalty = 1.1
    max_new_tokens = 100

    categorizer = TextCategorizer(model_name, precision=precision, temperature=temperature, do_sample=do_sample,
                                  repetition_penalty=repetition_penalty, max_new_tokens=max_new_tokens)
    result_df = categorizer.process_data(csv_path, user_prompt, num_samples, text_column, label_column)
    print(result_df['pred_label'])
