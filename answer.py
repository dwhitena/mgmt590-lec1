#Importing Libraries
import pandas as pd
from transformers.pipelines import pipeline
#Modeling (Changed the model to a different one from Hugging Face - Got the same answers for all 5 questions; Link - https://huggingface.co/deepset/xlm-roberta-large-squad2)
hg_comp = pipeline('question-answering', model="deepset/xlm-roberta-large-squad2", tokenizer="deepset/xlm-roberta-large-squad2")
#Reading File
data = pd.read_csv('examples.csv')
#Iterating Over Data
for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer = hg_comp({'question': question, 'context': context})['answer']
    print(question)
    print(answer)
