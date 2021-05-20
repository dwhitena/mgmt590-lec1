import pandas as pd
from transformers.pipelines import pipeline

hg_comp = pipeline('question-answering', model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased-distilled-squad")

data = pd.read_csv('/Users/vb/Documents/College/Production Scale Data Products/examples.csv')

hg_comp2 = pipeline(model="bert-large-cased-whole-word-masking-finetuned-squad", tokenizer="bert-large-cased-whole-word-masking-finetuned-squad", task="question-answering")

for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer = hg_comp2({'question': question, 'context': context})['answer']
    score = hg_comp2({'question': question, 'context': context})['score']
    print(question, end='   :   ')
    print(answer)
    print(score)

for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer = hg_comp({'question': question, 'context': context})['answer']
    score = hg_comp({'question': question, 'context': context})['score']
    print(question, end='   :   ')
    print(answer)
    print(score)
#This was very fun!
