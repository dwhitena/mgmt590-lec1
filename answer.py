import pandas as pd
from transformers.pipelines import pipeline

hg_comp = pipeline('question-answering', model="bert-large-cased-whole-word-masking-finetuned-squad", tokenizer="bert-large-cased-whole-word-masking-finetuned-squad")

data = pd.read_csv('Example_Data - Sheet1.csv')

for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer = hg_comp({'question': question, 'context': context})['answer']
    print('The answer to the question "' + question + '" is "' + answer + '"')

#Answers:
#The answer to the question "who won the match sri lanka or india?" is "India"
#The answer to the question "what was the prison called in the green mile?" is "Cold Mountain Penitentiary"
#The answer to the question "what is the widest highway in north america?" is "Highway 401"
#The answer to the question "what makes the center of an atom stable?" is "The closer an electron is to the nucleus, the greater the attractive force"
#The answer to the question "who did holly matthews play in waterloo rd?" is "Leigh-Ann Galloway"
