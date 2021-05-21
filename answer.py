# Importing Libraries

import pandas as pd
import gradio as gr
from transformers.pipelines import pipeline
from statistics import mean


# Importing data
data = pd.read_csv('Example_Data - Sheet1.csv')


# Checking data
data.head()


# Checking for null values
data.isna().sum()


# Average Length of Question and Contexts
lstLength_Que = [len(question) for question in data.question]
lstLength_Context = [len(context) for context in data.context]

print("Average length of questions is " + str(round(mean(lstLength_Que),2)))
print("Average length of conetxt is " + str(round(mean(lstLength_Context),2)))


#Implementing First Model
model1 = pipeline('question-answering', model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased-distilled-squad")


#Implementing Second Model
model2 = pipeline('question-answering', model="bert-large-uncased-whole-word-masking-finetuned-squad", tokenizer="bert-large-uncased-whole-word-masking-finetuned-squad")


#Finding answers using first Model

for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer = model1({'question': question, 'context': context})['answer']
    print(answer)


#Finding answers using second Model

for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer = model2({'question': question, 'context': context})['answer']
    print(answer)


# Building web user Interface

# Using first Model as Final model for interactive web interface

def getanswer(que: str, context: str):
    input = {
        "question": que, "context": context
    }
    return model1(input)['answer']

gr.Interface(fn=getanswer, inputs=["textbox", "text"], outputs= "text" ).launch()


