import pandas as pd
from transformers.pipelines import pipeline

hg_comp1 = pipeline('question-answering', model="bert-large-uncased-whole-word-masking-finetuned-squad", tokenizer="bert-large-uncased-whole-word-masking-finetuned-squad")

data = pd.read_csv('Example_Data - Sheet1.csv')

for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    print(question)
    answer1 = hg_comp({'question': question, 'context': context})['answer']
    score1 = hg_comp({'question': question, 'context': context})['score']
    print(answer1)
    print(score1)

hg_comp2 = pipeline('question-answering', model="distilbert-base-cased-distilled-squad", tokenizer="distilbert-base-cased-distilled-squad")

for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    print(question)
    answer2 = hg_comp2({'question': question, 'context': context})['answer']
    score2 = hg_comp2({'question': question, 'context': context})['score']
    print(answer2)
    print(score2)
