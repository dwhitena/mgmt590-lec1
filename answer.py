import pandas as pd
from transformers.pipelines import pipeline

hg_comp_multilingual = pipeline('question-answering', model="bert-base-multilingual-uncased", tokenizer="bert-base-multilingual-uncased")

hg_comp = pipeline('question-answering', model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased-distilled-squad")

data = pd.read_csv('examples.csv')

for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer_M = hg_comp_multilingual({'question': question, 'context': context})['answer']
    answer = hg_comp({'question': question, 'context': context})['answer']
    print(answer_M + "multilingual" + '/n' + answer + "Original")
