# mgmt590-lec1
## Question-Answering System
It is a model created with NLP which gives short answers to a question asked on the basis of the knowledge base given to the machine. 
![Alt Text](https://www.kdnuggets.com/wp-content/uploads/qa-system-similarity-1.png)
## Data
We have used examples.csv. The dataset has 3 columns and 5 rows.</br>
The columns are:
1. Question
2. Context - The model trains on this and gets the answer
3. Answer - This is just for reference not used while fitting the model
## Requirements 
1. Python 3
2. Transformers library installed </br>
```
pip install transformers
```
For Pytorch,
```
pip install transformers[torch]
```
## Models used
1. Cased Bert base model
```
model1 = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
tokenizer1 = AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')
modelBC = pipeline('question-answering', model = model1, tokenizer = tokenizer1)
```
2. Uncased Bert base model
```
tokenizer2 = BertTokenizer.from_pretrained('bert-base-uncased')
model2 = BertModel.from_pretrained("bert-base-uncased")
modelBC = pipeline('question-answering', model = model2, tokenizer = model2)
```

3. Uncased DistilBert base model
```
hg_comp = pipeline('question-answering', model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased-distilled-squad")
```

## Model Fitting
We have iterated over different rows to get the result for each of the question from the context column. 

```
for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer1 = modelBC({'question': question, 'context': context})['answer']
    print("Answer from Cased Bert base model: "answer1)
    answer2 = modelUBC({'question': question, 'context': context})['answer']
    print("Uncased Bert base model: "answer2)
    answer3 = hg_comp({'question': question, 'context': context})['answer']
    print("Answer from Uncased DistilBert base model: "answer3)
```
There are a lot of models available for question answering. so it is important to compare results from different models so we know which model will work the best for our dataset. As the size of the dataset is small, I have added the results of each of the model to see which model performs the best. 
