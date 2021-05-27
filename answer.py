#importing libraries
import pandas as pd
from transformers.pipelines import pipeline

#Use  huggingface transformers
hg_comp = pipeline('question-answering', model="mrm8488/bert-small-finetuned-squadv2", tokenizer="mrm8488/bert-small-finetuned-squadv2")

#Input the data
data = pd.read_csv('examples.csv')

#Answer the question
for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer = hg_comp({'question': question, 'context': context})['answer']
    print("The answer to question number",(idx+1),"is",answer)

#Use huggingface transformers
hg_comp = pipeline('question-answering', model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased-distilled-squad")

#Answer the question
for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer = hg_comp({'question': question, 'context': context})['answer']
    print(answer)
