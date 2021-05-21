
## Code Description
This is a readme file.
The script answer.py contains a python code to answer the question asked based on the information provided in the context.
The questions are contained in a csv file called examples.csv with questions under the column "questions" and context under the column "context".
Our code uses multiple models to provide answers and confidence scores for each answer.

## Model Description

** 1. deepset/roberta-base-squad2 **
Language model: roberta-base
Language: English
Downstream-task: Extractive QA
Training data: SQuAD 2.0
Eval data: SQuAD 2.0
Code: See example in FARM
Infrastructure: 4x Tesla v100

** 2. bert-large-uncased-whole-word-masking-finetuned-squad **
BERT is a transformers model pretrained on a large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it was pretrained with two objectives:

**Masked language modeling (MLM):** taking a sentence, the model randomly masks 15% of the words in the input then run the entire masked sentence through the model and has to predict the masked words. This is different from traditional recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like GPT which internally mask the future tokens. It allows the model to learn a bidirectional representation of the sentence.

**Next sentence prediction (NSP):** the models concatenates two masked sentences as inputs during pretraining. Sometimes they correspond to sentences that were next to each other in the original text, sometimes not. The model then has to predict if the two sentences were following each other or not.
This way, the model learns an inner representation of the English language that can then be used to extract features useful for downstream tasks: if you have a dataset of labeled sentences for instance, you can train a standard classifier using the features produced by the BERT model as inputs.

** 3. mrm8488/bert-tiny-finetuned-squadv2 **
The smaller BERT models are intended for environments with restricted computational resources. They can be fine-tuned in the same manner as the original BERT models. However, they are most effective in the context of knowledge distillation, where the fine-tuning labels are produced by a larger and more accurate teacher.