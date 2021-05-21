#Importing Libraries
import pandas as pd
from transformers.pipelines import pipeline

#Implementing Bert-Large-Uncased Model
hg_comp = pipeline('question-answering', model="bert-large-uncased", tokenizer="bert-large-uncased")

#Reading Data
data = pd.read_csv('examples.csv')

#Display Answer
for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer = hg_comp({'question': question, 'context': context})['answer']
    print("The answer to the question: \n" + question + "\nAns: " + answer)
