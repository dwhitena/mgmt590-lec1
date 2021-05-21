#importing
import pandas as pd   
from transformers.pipelines import pipeline

#original model
hg_comp = pipeline('question-answering', model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased-distilled-squad")
#improved model
hg_comp_new = pipeline('question-answering', model="bert-large-uncased-whole-word-masking-finetuned-squad", tokenizer="bert-large-uncased-whole-word-masking-finetuned-squad")

#reading the file
data = pd.read_csv('examples.csv')

#finding the answer to the question based on both the models
for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer = hg_comp({'question': question, 'context': context})['answer']
    answer_new = hg_comp_new({'question': question, 'context': context})['answer']
    print("Question: " + question)
    print('Answer (based on original model): ' + answer)
    print('Answer (based on improved model): ' + answer_new)


