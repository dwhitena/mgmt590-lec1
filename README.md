# Question-Answering API and Deployment on GCU

## Project Objective

This repository contains an implementation of the question-answering system using hugging face transformers with the help of SQLite database. Question Answering can be used in a variety of use cases. A very common one: Using it to navigate through complex knowledge bases or long documents ("search setting" ). The main goal of this project is applying multiple routes and endpoints and getting the answer using the model suggested on your local as well as on the cloud.In order to do this, we will be using a SQLite database to log all answered questions, contexts, the model used to generate the answer,timestamps when required and answers.


### Getting Started
To get started with this project:
 - We would require our local python environment eg:Jupyter, Pycharm etc.
 - A Dockerfile and requirements on our master repository. A yaml file to create a pipeline to deploy our code to GCP
 - Google cloud platform (GCP) in order to deploy the code using docker image


### List of all available routes/resources

For this project, we will be defining some routes using some HTTP methods.

#### HTTP Methods
HTTP methods are sometimes referred to as HTTP verbs. They are simply just different ways to communicate via HTTP. The ones that we will be using for this project are mentioned below:

- GET should be used for retrieving data from the API.
- POST should be used for creating new resources (i.e users, posts, taxonomies)
- PUT should be used for updating resources.
- DELETE should be used for deleting resources.

**List Available Models**

*This route allows a user to obtain a list of the models currently loaded into the server and available for inference*

Method and path:
```
GET /models
```

Expected Response Format:
```
[{
"name": "distilled-bert",
"tokenizer": "distilbert-base-uncased-distilled-squad",
"model": "distilbert-base-uncased-distilled-squad"},

{"name": "deepset-roberta",
"tokenizer": "deepset/roberta-base-squad2",
"model": "deepset/roberta-base-squad2"}
]
```

**Add a Model**

*This route allows a user to add a new model into the server and make it available for inference*

Method and path: 
```
PUT /models
```
Expected Request Body Format:
```
{
"name": "bert-tiny",
"tokenizer": "mrm8488/bert-tiny-5-finetuned-squadv2",
"model": "mrm8488/bert-tiny-5-finetuned-squadv2"
}
```
Expected Response Format (updated list of available models):

```
[{
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
}]
```

**Delete a Model**

*This route allows a user to delete an existing model on the server such that it is no longer available for inference*

Method and path:
```
DELETE /models?model=<model name>
```

Expected Response Format (updated list of available models):
```
[{
"name": "distilled-bert",
"tokenizer": "distilbert-base-uncased-distilled-squad",
"model": "distilbert-base-uncased-distilled-squad"
},
{
"name": "deepset-roberta",
"tokenizer": "deepset/roberta-base-squad2",
"model": "deepset/roberta-base-squad2"
}]
```

**Answer a Question**

*This route uses one of the available models to answer a question, given the context provided in the JSON payload*

Method and path:
```
POST /answer?model=<model name>
```
Query Parameters:

  - model name (optional) - The name of the model to be used in answering the question. If no model name is provided use a default model.
 
  
Expected Request Body Format: 
```
{
"question": "who did holly matthews play in waterloo rd?",
"context": "She attended the British drama school East 15 in 2005, and left after winning a high-profile role in the BBC drama Waterloo Road, playing the bully Leigh-Ann Galloway.[6] Since that role, Matthews has continued to act in BBC's Doctors, playing Connie Whitfield; in ITV's The Bill playing drug addict Josie Clarke; and she was back in the BBC soap Doctors in 2009, playing Tansy Flack."
} 
```

Expected Response Format:
```
{
"timestamp": 1621602784,
"model": "deepset-roberta",
"answer": "Leigh-Ann Galloway",
"question": "who did holly matthews play in waterloo rd?",
"context": "She attended the British drama school East 15 in 2005, and left after winning a high-profile role in the BBC drama Waterloo Road, playing the bully Leigh-Ann Galloway.[6] Since that role, Matthews has continued to act in BBC's Doctors, playing Connie Whitfield; in ITV's The Bill playing drug addict Josie Clarke; and she was back in the BBC soap Doctors in 2009, playing Tansy Flack."
}
```

**List Recently Answered Questions**
 
*This route returns recently answered questions*

Method and path:
```
GET /answer?model=<model name>&start=<start timestamp>&end=<end timestamp>  
```
Query Parameters:
- model name (optional) - Filter the results by providing a certain model name, such
that the results only include answered questions that were answered using the provided
model.
- start timestamp (required) - The starting timestamp, such that answers to questions
prior to this timestamp won't be returned.
- end timestamp (required) - The ending timestamp, such that answers to questions
after this timestamp won't be returned

Expected Response Format (updated list of available models):
```
[{
"timestamp": 1621602784,
"model": "deepset-roberta",
"answer": "Leigh-Ann Galloway",
"question": "who did holly matthews play in waterloo rd?",
"context": "She attended the British drama school East 15 in 2005, and left after winning a high-profile role in the BBC drama Waterloo Road, playing the bully Leigh-Ann Galloway.[6] Since that role, Matthews has continued to act in BBC's Doctors, playing Connie Whitfield; in ITV's The Bill playing drug addict Josie Clarke; and she was back in the BBC soap Doctors in 2009, playing Tansy Flack."
},
{
"timestamp": 1621602930,
"model": "distilled-bert",
"answer": "Travis Pastrana",
"question": "who did the first double backflip on a dirt bike?",
"context": "2006 brought footage of Travis Pastrana completing a double backflip on an uphill/sand setup on his popular /"Nitro
Circus/" Freestyle Motocross movies. On August 4, 2006, at X Games 12 in Los Angeles, he became the first rider to land a double backflip in competition. Having landed another trick that many had considered impossible, he vowed never to do it again."
}]
```
  

## About API and launching the API 

The term API, short for Application Programming Interface, refers to a part of a computer program designed to be used or manipulated by another program, as opposed to an interface designed to be used or manipulated by a human. Computer programs frequently need to communicate amongst themselves or with the underlying operating system, and APIs are one way they do it.

The main anatomy behind a REST API is composed of three key items: (1) the url& endpoint, (2) the method, and (3) the data. When a client makes an HTTP request against an API in order to retrieve data, the first item that must be designed is the URL. The URL generally encompasses the sites domain, a series of directory hierarchies, and finally the endpoint 

**Routes:** Routes in the REST API are represented by URIs.

**Endpoints:** Endpoints are functions available through the API. This can be things like retrieving the API index, updating a post, or deleting a comment. Endpoints perform a specific function, taking some number of parameters and return data to the client. 

A route is the “name” you use to access endpoints, used in the URL. A route can have multiple endpoints associated with it, and which is used depends on the HTTP verb.

Launching the API can be done by Flask. The application can be run using Flask’s run functionality *app.run()* .We have to first import flask and then run the code for running the API. One very crucial thing that happens during launching is to decide the address of host. We can also host the API in our own local machine. We can also use multiple online service to host.

### Dependencies
Flask, transformers and torch are the dependencies required for this project. They have been mentioned in the requirements.txt in the repository. The flask dependency consists of sqlite3.

### How to build and run the API locally via Docker or Flask

The app will locally run through the link on port 8080 and can be accessed through postman. All the routes will work as it is there.

In order to duplicate the project outcome, one can follow the below steps:
1. Git clone the same repository into your local
2. Pip install all the dependencies required for the project (eg:flask, tensorflow etc.)
3. Run the code on your system and check the results using postman
4. If you wish to deploy it on the cloud, create a Dockerfile and requirements.txt in your repository
5. Setup your cloud platform and deploy the code on the system to receive the api link
