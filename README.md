# Text Classification in the Age of Large Language Models

## Abstract

Recent advances in Large Language Models (LLMs) have revealed exceptional capability in tasks such as question answering, summarization, and machine translation. The assumption that larger models produce superior outcomes has prompted the release of LLMs with billions of parameters. These models, while powerful, are restricted because of hardware limitations and accessibility issues for the broader machine learning community and use. To mitigate these issues, efficient adaptation techniques like quantization, prefix tuning, and low-rank adaptation have been developed, enabling precise fine-tuning to specific applications. Furthermore, innovations such as instruction tuning and alignment training allow LLMs to perform tasks through human-written instructions known as prompts without further adaptation. Most of these advances have focused on improving results in the text generation space and less attention has been paid to whether these advances generate the same value for text classification in the context of LLMs.
This research  investigates the influence of model scale, pretraining objectives, and quantization for text classification tasks. Our findings demonstrate that larger models do not inherently lead to better outcomes. We demonstrate that fine-tuning significantly outperforms prompting techniques, and that an increase in model parameters does not always enhance classification performance. Interestingly, smaller models often yield results that are either comparable to or better than those of larger models in classification tasks.

## Authors

- **Paul Trust**
  - **Institution:** [University College Cork](https://www.ucc.ie/en/)
  - **Email:** [120222601@umail.ucc.ie](mailto:120222601@umail.ucc.ie)

