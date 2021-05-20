import pandas as pd
from transformers.pipelines import pipeline
    
hg_comp = pipeline('question-answering', model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")

data = pd.read_csv('Example_Data1.csv')
i= 0 
for idx, row in data.iterrows():
    i = i+1
    context = row['context']
    question = row['question']
    answer = hg_comp({'question': question, 'context': context})['answer']

    print('Q.',i,question)
    print('Ans.',answer)
