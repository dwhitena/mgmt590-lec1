#Importing Libraries
import pandas as pd
import numpy as np
import glob

# Importing Pipeline from transformer library
from transformers.pipelines import pipeline

# Calling the pipeline function by giving model argument as "bert-base-multilingual-uncased" and tokenizer argument as "bert-base-multilingual-uncased"
hg_BERT = pipeline('question-answering', model="bert-base-multilingual-uncased", tokenizer="bert-base-multilingual-uncased")

# Loading the dataset
data = pd.DataFrame()
for f in glob.glob(r"C:\Users\abhin\OneDrive\Desktop\qachat\*.csv"):
    df = pd.read_csv(f)
    data = data.append(df,ignore_index=True)

#Creating a empty list to store the answers
answer_li = []
##Lopping over the dataframe df and feeding the question and context as an input into the pre trained model to generate respective answers 
##and finally storing those answers into a list
for i, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer = hg_BERT({'question': question, 'context': context})['answer']
    answer_li.append(answer)

# Creating a new dataframe to store the data from Context, Question and Answers in one place, for a better visualization
df_new = pd.DataFrame()
df_new['context'] = data['context']
df_new['question'] = data['question']
df_new['answer'] = answer_li
df_new
