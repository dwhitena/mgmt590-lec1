# importing the necessary libraries
import pandas as pd
from transformers.pipelines import pipeline

#Choosing the optimum hyper-paramters for the model
hg_comp1 = pipeline('question-answering', model="xlnet-base-cased", tokenizer="distilbert-base-uncased-distilled-squad")

#Creating one more model for comparision
hg_comp2 = pipeline('question-answering', model="xlm-mlm-en-2048", tokenizer="distilbert-base-uncased-distilled-squad")

#Loading the dataset
data = pd.read_csv('examples.csv')

#Model outputs
ans1 = []
ans2 = []
for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer1 = hg_comp1({'question': question, 'context': context})['answer']
    answer2 = hg_comp2({'question': question, 'context': context})['answer']
    ans1.append(answer1)
    ans2.append(answer2)
    print("Model 1 Answer: " + answer1)
    print("Model 2 Answer: " + answer2)
    
#Appending the output as columns in the original dataset to compare
data["Model 1"] = ans1
data["Model 2"] = ans2
