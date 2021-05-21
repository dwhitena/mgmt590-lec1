# Importing libraries
import pandas as pd
from transformers.pipelines import pipeline

hg_comp = pipeline('question-answering', model="madlag/bert-base-uncased-squadv1-x2.32-f86.6-d15-hybrid-v1",
    tokenizer="madlag/bert-base-uncased-squadv1-x1.16-f88.1-d8-unstruct-v1")

# Reading CSV file
data = pd.read_csv('examples.csv')

# Looping through each question and finding the answer using the pipeline
for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer = hg_comp({'question': question, 'context': context})['answer']
    print('Question: ' +question+ '\n' + 'Answer: ' + answer)
    print('\n')


