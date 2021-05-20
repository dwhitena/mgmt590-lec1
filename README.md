# Question Answering Model implementation

In this project, the objective is to display the correct answer to a question asked by a user. An example.csv file is used, containing questions and a given context to derive answers to those questions.
We have used 2 pretrained models from Hugging Face Transformers, fine tuned for a question answering task, namely:
- **bert-large-cased-whole-word-masking-finetuned-squad**
  This model was trained using Whole Word Masking, where all of the tokens corresponding to a word are masked at once. After pre-training, this model was fine-tuned on the SQuAD dataset with one of our fine-tuning scripts. See below for more information regarding this fine-tuning.
  
- **distilbert-base-uncased-distilled-squad**
  This model is a fine-tune checkpoint of DistilBERT-base-uncased, fine-tuned using knowledge distillation on SQuAD v1.1.

These models have been observed to provide correct answers to the questions and can be compared and used based on preferred answers and performance.


