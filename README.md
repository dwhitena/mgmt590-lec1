# Question Aswering API

## General Information about API

This is a Question Answering API which answers the question from a context. It provides option to choose the model of choice to get answer for the question. Other functions include getting the present models and the answers provided by them with timestamp.

## Available URLS
There are 5 different functionalities which are facilitating in their own way. They are providing multiple functionalities.They are:a

1) GET /models
2) PUT /models
3) DELETE /models?model=<model name>
4) POST /answer?model=<model name>
5) GET /answer?model=<model name>&start=<start timestamp>&end=<end timestamp>
 

First route is to get the name of all the available models at present. 
  
### Route: 
  ```
  GET /models
  ```
### Expected Output
 [
{
"name": "distilled-bert",
"tokenizer": "distilbert-base-uncased-distilled-squad",
"model": "distilbert-base-uncased-distilled-squad"
},
{
"name": "deepset-roberta",
"tokenizer": "deepset/roberta-base-squad2",
"model": "deepset/roberta-base-squad2"
}
]
Second routes is used to enter a new model in the available lists of model. Once API receives PUT request to this handler, the name of model is extracted from request and that particular model is made available to use in future. 
  
### Route: 
  ```
  PUT /models
  ```
### JSON Body:
 ```
{
"name": "bert-tiny",
"tokenizer": "mrm8488/bert-tiny-5-finetuned-squadv2",
"model": "mrm8488/bert-tiny-5-finetuned-squadv2"
}
```
### Expected Output
 [
{
"name": "distilled-bert",
"tokenizer": "distilbert-base-uncased-distilled-squad",
"model": "distilbert-base-uncased-distilled-squad"
},
{
"name": "deepset-roberta",
"tokenizer": "deepset/roberta-base-squad2",
"model": "deepset/roberta-base-squad2"
},
{
"name": "bert-tiny",
"tokenizer": "mrm8488/bert-tiny-5-finetuned-squadv2",
"model": "mrm8488/bert-tiny-5-finetuned-squadv2"
}
]
  
Third route is to delete a model from the dataset. It extracts the name of model from DELETE request and deletes that particular model.
 
### Route: 
  ```
  DELETE /models?model=<model name>
  ```
 ### Expected Output
[
{
"name": "distilled-bert",
"tokenizer": "distilbert-base-uncased-distilled-squad",
"model": "distilbert-base-uncased-distilled-squad"
},
{
"name": "deepset-roberta",
"tokenizer": "deepset/roberta-base-squad2",
"model": "deepset/roberta-base-squad2"
}
]
  
Fourth route is the most important route. It answers the question on basis of context. It takes a post request and extract the Question and context from the body of request. An option is also there to choose the model. In this case API will use the same model to answer the question. If no model name is given then API uses the default model and predict the answer. Once the answer is predicte and returned to client, the question, context, answer , name of the model used and time stamp is noted to maintain the record of past data. 

### Route: 
  ```
  POST /answer?model=<model name>
  ```
### JSON BODY:
```
  {
"question": "who did holly matthews play in waterloo rd?",
"context": "She attended the British drama school East 15 in 2005,
and left after winning a high-profile role in the BBC drama Waterloo
Road, playing the bully Leigh-Ann Galloway.[6] Since that role,
Matthews has continued to act in BBC's Doctors, playing Connie
Whitfield; in ITV's The Bill playing drug addict Josie Clarke; and
she was back in the BBC soap Doctors in 2009, playing Tansy Flack."
}
 ```
### Expected Output
 {
"timestamp": 1621602784,
"model": "deepset-roberta",
"answer": "Leigh-Ann Galloway",
"question": "who did holly matthews play in waterloo rd?",
"context": "She attended the British drama school East 15 in 2005,
and left after winning a high-profile role in the BBC drama Waterloo
Road, playing the bully Leigh-Ann Galloway.[6] Since that role,
Matthews has continued to act in BBC's Doctors, playing Connie
Whitfield; in ITV's The Bill playing drug addict Josie Clarke; and
she was back in the BBC soap Doctors in 2009, playing Tansy Flack."
}
 
Fifth funtionality is to provide the history of Questions answered. It provides an option to select a specific model and know the details of its activity. In this case the name of model is extracted and activity of that model is returned. That is the question , context and answer along with time stamp of answering is returned. There is also option to select time frame . the start and the end time frame if selected, the API returns the records between that provided window. If there is no start and end time , API returns the records of all the instance wher a model was used to answetr a qyestion.In same way if the model name is not provided in request, the record for all models are returned.

### Route:
  ```
  GET /answer?model=<model name>&start=<start timestamp>&end=<end timestamp>
  ```
### Expected Output
 [
{
"timestamp": 1621602784,
"model": "deepset-roberta",
"answer": "Leigh-Ann Galloway",
"question": "who did holly matthews play in waterloo rd?",
"context": "She attended the British drama school East 15 in
2005, and left after winning a high-profile role in the BBC drama
Waterloo Road, playing the bully Leigh-Ann Galloway.[6] Since that
role, Matthews has continued to act in BBC's Doctors, playing Connie
Whitfield; in ITV's The Bill playing drug addict Josie Clarke; and
she was back in the BBC soap Doctors in 2009, playing Tansy Flack."
},
{
"timestamp": 1621602930,
"model": "distilled-bert",
"answer": "Travis Pastrana",
"question": "who did the first double backflip on a dirt bike?",
"context": "2006 brought footage of Travis Pastrana completing a
double backflip on an uphill/sand setup on his popular /"Nitro
Circus/" Freestyle Motocross movies. On August 4, 2006, at X Games 12
in Los Angeles, he became the first rider to land a double backflip
in competition. Having landed another trick that many had considered
impossible, he vowed never to do it again."
}
]

### Dependencies
Below libraries are required for tghis app to work:
1) Flask version 1.1.2
2) Pytorch version 1.8.1
3) Transformers version 46.1

Below softwares are required:
1) Spyder IDE
2) Postman
 
## Launching the API
### Where the API can be located (the base URL):  https://answer-api-jlzk3jod5q-uc.a.run.app

The app will run from the above link through Postman app. All the required routes need to be be given alogwith the required parameters and JSON body inputs.
  
## How to build and run the API locally via Docker or Flask
The app will locally run thorugh localhost link on port 8080 and can be accessed through postman.All the routes will work as it is there.
