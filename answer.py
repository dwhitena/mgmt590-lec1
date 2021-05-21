import pandas as pd
from transformers.pipelines import pipeline
from transformers import BertForQuestionAnswering, AutoTokenizer
from transformers import BertTokenizer, BertModel

# Model 1 - Cased Bert base model
model1 = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
tokenizer1 = AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')
modelBC = pipeline('question-answering', model = model1, tokenizer = tokenizer1)


#Model 2 - Uncased Bert base model
tokenizer2 = BertTokenizer.from_pretrained('bert-base-uncased')
model2 = BertModel.from_pretrained("bert-base-uncased")
modelUBC = pipeline('question-answering', model = model2, tokenizer = model2)

#Model 3 - Uncased DistilBert base model
hg_comp = pipeline('question-answering', model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased-distilled-squad")

#Reading the File
data = pd.read_csv('examples.csv')

#Preview of Data uploaded
print(data.head())

#Iterating through the dataset to generate answers on the basis of the context
for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer1 = modelBC({'question': question, 'context': context})['answer']
    print("Answer from Cased Bert base model: "answer1)
    answer2 = modelUBC({'question': question, 'context': context})['answer']
    print("Uncased Bert base model: "answer2)
    answer3 = hg_comp({'question': question, 'context': context})['answer']
    print("Answer from Uncased DistilBert base model: "answer3)
