import pandas as pd
import glob

from transformers.pipelines import pipeline

hg_comp = pipeline('question-answering', model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased-distilled-squad")

data = []
for filename in glob.glob('*.csv'):
    im = pd.read_csv(filename)
    data.append(im)
for data1 in data:
    for idx, row in data1.iterrows():
        context = row['context']
        question = row['question']
        answer = hg_comp({'question': question, 'context': context})['answer']
        print(answer)


    
    
