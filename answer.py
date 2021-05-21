# Importing Necessary Libraries
import pandas as pd
import glob
import gradio as gr
from transformers.pipelines import pipeline

# Replaced old model with a new more efficient model 
model = pipeline('question-answering', model="bert-large-uncased-whole-word-masking-finetuned-squad", tokenizer="bert-large-uncased-whole-word-masking-finetuned-squad")

#Reading output file using for loop and glob function 
data = []
for filename in glob.glob('*.csv'):
    im = pd.read_csv('Example_Data - Sheet1.csv')
    data.append(im)
for data1 in data:
    for idx, row in data1.iterrows():
        context = row['context']
        question = row['question']
        answer = model({'question': question, 'context': context})['answer']
        print('Q.',question)
        print('Ans.',answer)

## Building web user Interface

def getanswer(que: str, context: str):
    input = {
        "question": que, "context": context
    }
    return model(input)['answer']

gr.Interface(fn=getanswer, inputs=["textbox", "text"], outputs= "text" ).launch()
        

