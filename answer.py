#Importing Libraries
import pandas as pd
from transformers.pipelines import pipeline

#Applying the model; link: https://huggingface.co/deepset/roberta-base-squad2)
nlp = pipeline('question-answering', model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")
#Reading csv
data = pd.read_csv('examples.csv')

#Iterating
for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer = nlp({'question': question, 'context': context})['answer']
    print(question)
    print(answer)
    
   
