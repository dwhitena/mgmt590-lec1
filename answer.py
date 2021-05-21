import warnings

import pandas as pd
from transformers.pipelines import pipeline

# Defining model for given objective (question-answering)
hg_comp = pipeline('question-answering', model="distilbert-base-uncased-distilled-squad",
                   tokenizer="distilbert-base-uncased-distilled-squad")
hg_comp2 = pipeline('question-answering', model="bert-large-uncased-whole-word-masking-finetuned-squad",
                   tokenizer="bert-large-uncased-whole-word-masking-finetuned-squad")
# Import file with question and context
data = pd.read_csv('examples.csv')

# Loop to iterate over each row and find answers
for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    # distilbert-base-uncased-distilled-squad
    answer = hg_comp({'question': question, 'context': context})['answer']
    # bert-large-uncased-whole-word-masking-finetuned-squad
    answer2 = hg_comp2({'question': question, 'context': context})['answer']
    print(question)
    print(answer)
    print(answer2)
    print("\n")
