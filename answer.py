import pandas as pd
from transformers.pipelines import pipeline

hg_comp = pipeline('question-answering', model='bert-base-cased-squad2', tokenizer='distilbert-base-uncased-squad')

data = pd.read_csv('examples.csv')

for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer = hg_comp({'question': question, 'context': context})['answer']
    print(answer)

