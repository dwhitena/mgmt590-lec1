#Importing Libraries
import pandas as pd
import numpy as np

# Importing Pipeline from transformer library
from transformers.pipelines import pipeline

# Calling the pipeline function by giving model argument as "bert-base-multilingual-cased" and tokenizer argument as "distilbert-base-uncased-distilled-squad"
hg_BERT = pipeline('question-answering', model="bert-base-multilingual-cased", tokenizer="distilbert-base-uncased-distilled-squad")

# Loading the dataset
data = pd.read_csv('examples.csv')

#Creating a empty list to store the answers
answer_li = []
##Lopping over the dataframe df and feeding the question and context as an input into the pre trained model to generate respective answers 
##and finally storing those answers into a list
for i, row in df.iterrows():
    context = row['context']
    question = row['question']
    answer = hg_BERT({'question': question, 'context': context})['answer']
    answer_li.append(answer)

# Creating a new dataframe to store the data from Context, Question and Answers
df_new = pd.DataFrame()
df_new['context'] = df['context']
df_new['question'] = df['question']
df_new['answer'] = answer_li
df_new
