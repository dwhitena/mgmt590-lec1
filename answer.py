import pandas as pd
from transformers.pipelines import pipeline

#importing the file using pandas library
data = pd.read_csv('Example_Data - Sheet1.csv')

#using transformer model bert large uncased masking to predict the answer to our question and also getting the score of the answer
hg_comp = pipeline('question-answering', model="bert-large-uncased-whole-word-masking-finetuned-squad", tokenizer="bert-large-uncased-whole-word-masking-finetuned-squad")
answer = []
score = []
questions = []
for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    questions.append(question)
    answer.append(hg_comp({'question': question, 'context': context})['answer'])
    score.append(hg_comp({'question': question, 'context': context})['score'])
print("")
print ("                                      ####### MODEL - BERT LARGE UNCASED WHOLE WORD MASKING FINE TUNED #######")

#printing the question, answer and score in the following format for the executed model
# QUESTION             | ANSWER   | SCORE
# What is this course? | MGMT 590 | 0.9565623 

titles = ['QUESTION','ANSWER', 'SCORE']

row_format = '{:<50}|{:<80}|{:<30}'
print (row_format.format(*titles))
for item in zip(questions, answer, score):
    print (row_format.format(*item))


#using transformer model distilbert base uncased to predict the answer to our question and also getting the score of the answer  
hg_comp_1 = pipeline('question-answering', model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased-distilled-squad")
answer_1 = []
score_1 = []
questions_1 = []
for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    questions_1.append(question)
    answer_1 .append(hg_comp_1({'question': question, 'context': context})['answer'])
    score_1.append(hg_comp_1({'question': question, 'context': context})['score'])
print("")
print ("                                      ####### MODEL - DISTILBERT BASE UNCASED DISTILLED #######")

#printing the question, answer and score in the following format for the executed model
# QUESTION             | ANSWER   | SCORE
# What is this course? | MGMT 590 | 0.9565623 

row_format = '{:<50}|{:<80}|{:<30}'
print (row_format.format(*titles))
for item in zip(questions_1, answer_1, score_1):
    print (row_format.format(*item))

#using transformer model bert based modified by csarron to predict the answer to our question and also getting the score of the answer  
hg_comp_2 = pipeline('question-answering', model="csarron/bert-base-uncased-squad-v1", tokenizer="csarron/bert-base-uncased-squad-v1")
answer_2 = []
score_2 = []
questions_2 = []
for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    questions_2.append(question)
    answer_2.append(hg_comp_2({'question': question, 'context': context})['answer'])
    score_2.append(hg_comp_2({'question': question, 'context': context})['score'])

print("") 
print ("                                      ####### MODEL - CSARRON's BERT BASE UNCASED #######")
#printing the question, answer and score in the following format for the executed model
# QUESTION             | ANSWER   | SCORE
# What is this course? | MGMT 590 | 0.9565623 
row_format = '{:<50}|{:<80}|{:<30}'
print (row_format.format(*titles))
for item in zip(questions_2, answer_2, score_2):
    print (row_format.format(*item))
