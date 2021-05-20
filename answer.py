#Importing libraries
import pandas as pd
from transformers.pipelines import pipeline

#Using pertained models
hg_comp = pipeline('question-answering', model="bert-large-cased-whole-word-masking-finetuned-squad", tokenizer="bert-large-cased-whole-word-masking-finetuned-squad")
hg_comp_1 = pipeline('question-answering', model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased-distilled-squad")


#Reading the csv into a dataframe
data = pd.read_csv('examples.csv')

print("Answers using Model 1:\n")
for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer = hg_comp({'question': question, 'context': context})['answer']
    print("Q", (idx+1), question, ": \n Ans. ", answer)

print("Answers using Model 2:\n")
for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer = hg_comp_1({'question': question, 'context': context})['answer']
    print("Q", (idx+1), question, ": \n Ans. ", answer)
    
