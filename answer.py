#importing library
import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers.pipelines import pipeline
import torch

#Modelling
Tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
Model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

comp = pipeline('question-answering', model= Model, tokenizer = Tokenizer)

data = pd.read_csv('examples.csv')

for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer = comp({'question': question, 'context': context})['answer']
    print(question)
    print(answer)
