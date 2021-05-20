#importing libraries
import pandas as pd
from transformers.pipelines import pipeline

#importing model
hg_comp = pipeline('question-answering', model="etalab-ia/camembert-base-squadFR-fquad-piaf", tokenizer="etalab-ia/camembert-base-squadFR-fquad-piaf")

#importing data
data = pd.read_csv('C:/Users/gupta/Downloads/Example_Data - Sheet1.csv')

#finding out the answer to the questions
for idx, row in data.iterrows():
    context = row['context']
    print (row['question'])
    question = row['question']
    answer = hg_comp({'question': question, 'context': context})['answer']
    print(answer)
