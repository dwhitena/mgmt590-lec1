import pandas as pd
from transformers.pipelines import pipeline

hg_comp = pipeline('question-answering', model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased-distilled-squad")
hg_comp_bert = pipeline('question-answering', model="distilbert-base-uncased-distilled-squad", tokenizer='bert-base-cased')


data = pd.read_csv('examples.csv')

for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer = hg_comp({'question': question, 'context': context})['answer']
    answer_bert = hg_comp({'question': question, 'context': context})['answer']
    print(answer)
    print(answer_bert + " Bert base")
    
