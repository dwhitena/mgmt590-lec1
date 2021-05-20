# Importing Libraries
import pandas as pd
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Importing Data
data = pd.read_csv('examples.csv')

# Modelling
model_name = "deepset/roberta-base-squad2"

nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

# Iterating over each row
for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer = nlp({'question': question, 'context': context})['answer']
    print(question)
    print(answer)
