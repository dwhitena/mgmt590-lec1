#importing libraries
#import pandas as pd
import sqlite3
from transformers.pipelines import pipeline
from flask import Flask
from flask import request
from flask import jsonify
#import mysql.connector
from flask import json
import time
import os



app = Flask(__name__)
conn = sqlite3.connect('database.db')

listOfTables = conn.execute(
    """SELECT name FROM sqlite_master WHERE type='table'
    AND name='models'; """).fetchall()

if listOfTables==[]:
    cur = conn.cursor()
    conn.execute('create table models (name varchar(100),model varchar(100),tokenizer varchar(100))')
    sql_insert_query1 = "insert into models values(" + "'" + "distilled-bert" + "'" + "," \
                        + "'" + "distilbert-base-uncased-distilled-squad" + "'" + "," + "'" + \
                        "distilbert-base-uncased-distilled-squad" + "'" + ")"
    sql_insert_query2 = "insert into models values(" + "'" + "deepset-roberta" + "'" + "," \
                        + "'" + "deepset/roberta-base-squad2" + "'" + "," + "'" + \
                        "deepset/roberta-base-squad2" + "'" + ")"
    sql_insert_query3 = "insert into models values(" + "'" + "bert-tiny" + "'" + "," \
                        + "'" + "mrm8488/bert-tiny-5-finetuned-squadv2" + "'" + "," + "'" + \
                        "mrm8488/bert-tiny-5-finetuned-squadv2" + "'" + ")"
    cur.execute(sql_insert_query1)
    cur.execute(sql_insert_query2)
    cur.execute(sql_insert_query3)
    conn.commit()


else:
    print("table exists")


@app.route("/models", methods=['GET', "PUT", "DELETE"])
def models():
    if request.method == 'GET':
        conn1 = sqlite3.connect('database.db')
        sql_select_query = "select Distinct * from models"
        cursor = conn1.cursor()
        cursor.execute(sql_select_query)
        records = cursor.fetchall()
        list1 = []

        for record in records:
            out = {
                "name": record[0],
                "tokenizer": record[2],
                "model": record[1]
            }
            list1.append(out)
        return json.jsonify(list1)


    if request.method=="PUT":
        data = request.json
        conn1 = sqlite3.connect('database.db')

        sql_insert_query = "insert into models values(" + "'" + data['name']+"'" + "," \
                           +"'"+ data['model'] + "'" + "," + "'" + data["tokenizer"]+"'" + ")"
        cursor = conn1.cursor()
        cursor.execute(sql_insert_query)
        conn1.commit()

        sql_select_query = "select Distinct * from models"
        cursor.execute(sql_select_query)
        records = cursor.fetchall()
        list1 = []

        for record in records:
            out = {
                "name": record[0],
                "tokenizer": record[2],
                "model": record[1]
            }
            list1.append(out)
        return json.jsonify(list1)


    if request.method == "DELETE":
        conn1 = sqlite3.connect('database.db')
        model = request.args.get('model')
        sql_delete_query = "delete from models where" + "`" + "name" + "`" + "=" + "'" + str(model) + "'"
        cursor = conn1.cursor()
        cursor.execute(sql_delete_query)
        conn1.commit()

        sql_select_query = "select Distinct * from models"
        cursor = conn1.cursor()
        cursor.execute(sql_select_query)
        records = cursor.fetchall()
        list1 = []

        for record in records:
            out = {
                "name": record[0],
                "tokenizer": record[2],
                "model": record[1]
            }
            list1.append(out)
        return json.jsonify(list1)

conn.execute('CREATE TABLE if not exists answered (timestamp DATETIME, model TEXT, answer TEXT, question TEXT, context TEXT)')

@app.route("/answer", methods=['POST','GET'])
def answer():
    if request.method == 'POST':
        model_name = request.args.get('model')
        data = request.json
        conn1 = sqlite3.connect('database.db')
        sql_post_query = "select * from models where name =?"
        cursor = conn1.cursor()
        cursor.execute(sql_post_query,(model_name,))
        row = cursor.fetchone()

    # Import model
        hg_comp = pipeline('question-answering', model=row[1],tokenizer= row[2])
    # Answer the answer
        answer = hg_comp({'question': data['question'], 'context': data['context']})['answer']
        timestamp = int(time.time())
        sql_insertanswer_query = "insert into answered (timestamp, model, answer,question,context) values(?,?,?,?,?)"
        cursor.execute(sql_insertanswer_query,  (timestamp,model_name,answer,data['question'],data['context']))
        conn1.commit()
        out = {
            "timestamp": timestamp,
            "model": model_name,
            "answer": answer,
            "question": data['question'],
            "context": data['context']
        }
        return jsonify(out)

    if request.method == 'GET':
        start = request.args.get('start')
        end = request.args.get('end')
        conn1 = sqlite3.connect('database.db')
        cursor = conn1.cursor()
        query = " select * from answered where timestamp > ? and timestamp < ?"
        params =[start,end]
        if 'model' in request.args:
            model_name = request.args.get('model')
            query += "and model = ?"
            params += model_name
        cursor.execute(query,params)
        rows = cursor.fetchall()
        list1 = []
        for row in rows:
            out = {
                "timestamp": row[0],
                "answer":row[1],
                "model":row[2],
                "question":row[3],
                "context":row[4]
            }
            list1.append(out)
        return(jsonify(list1))


# Run if running "python answer.py"
if __name__ == '__main__':
    # Run our Flask app and start listening for requests!
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)), threaded=True)

