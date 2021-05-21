import pandas as pd
from transformers.pipelines import pipeline

model = "madlag/bert-base-uncased-squadv1-x2.32-f86.6-d15-hybrid-v1"
tokenizer = "madlag/bert-base-uncased-squadv1-x2.32-f86.6-d15-hybrid-v1"

hg_comp = pipeline('question-answering',
                   model=model,
                   tokenizer=tokenizer
                   )

path_str = "C:/Users/saite/Desktop/Masters/1.Purdue/Course Material/3.Sum Mod 1/MGMT 590 PSD/2.Assignments/examples.csv"

data = pd.read_csv(path_str)

for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer = hg_comp({'question': question, 'context': context})['answer']
    print('Question',idx+1,": ",question)
    print("Answer",idx+1,": ",answer)