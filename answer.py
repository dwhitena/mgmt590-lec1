import pandas as pd
from transformers.pipelines import pipeline

hg_comp1 = pipeline('question-answering', model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased-distilled-squad")
hg_comp2 = pipeline('question-answering', model="bert-base-uncased", tokenizer="bert-base-uncased")

data = pd.read_csv('examples.csv')

for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer1 = hg_comp1({'question': question, 'context': context})['answer']
    answer2 = hg_comp2({'question': question, 'context': context})['answer']
    data.at[idx,'Answer1'] = answer1
    data.at[idx,'Answer2'] = answer2
    
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer_start_scores, answer_end_scores = model(**inputs,return_dict=False)

    answer_start = torch.argmax(answer_start_scores)  # Get the most likely beginning of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score

    answer3 = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    data.at[idx,'Answer3'] = answer3

data.to_csv("Sample_answers.csv",index=False)
print(data)