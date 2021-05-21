import pandas as pd
from transformers.pipelines import pipeline

hg_comp = pipeline('question-answering', model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")

data = pd.read_csv('examples.csv')

for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer = hg_comp({'question': question, 'context': context})['answer']
    print("Q", (idx+1), question, ": \n Ans. ", answer + "\n")
