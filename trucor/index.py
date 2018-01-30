import sqlite3
import json
import pandas as pd
from sqlite3 import Error
from flask import Flask, jsonify
from flask import g
from flask import request
from datetime import datetime as dt
from trucor.ekg_analytics import sample_collection as sc
from trucor.ekg_analytics import ekgtemplate as et

app = Flask(__name__)

DATABASE = 'blueberry.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

def init_db():
    db = get_db()
    with app.open_resource('trucor_structure.sql', mode='r') as f:
        db.cursor().executescript(f.read())
    db.commit()

@app.cli.command('initdb')
def initdb_command():
    init_db()
    print('Initialized the database.')

@app.route('/')
def index():
    return "Welcome to truCOR! Raspberry Pi based biometric authentication platform."

if __name__ == '__main__':
    app.run(debug=True)

@app.route('/api/v1.0/users', methods=['GET'])
def list_users(json_str=True):
    db = get_db()
    cursor = db.cursor()
    rows = cursor.execute('''SELECT * FROM users''').fetchall()
    all_users = [{'Profile':{   'ACCOUNT ID': user[0],
                                'LAST NAME': user[1],
                                'FIRST NAME': user[2],
                                'EMAIL': user[4],
                                'USERNAME': user[3]}} for user in rows]
    return jsonify(all_users)
    

@app.route('/api/v1.0/users', methods=['POST'])
def enroll_user():
    try:
        if not request.json or not 'LAST NAME' in request.json:
            abort(400)
        db = get_db()
        cursor = db.cursor()
        lastname1 = request.json['LAST NAME']
        firstname1 = request.json['FIRST NAME']
        username1 = request.json['USERNAME']
        email1 = request.json['EMAIL']
        cursor.execute('''SELECT COUNT(*) FROM users''')
        total_accts = cursor.fetchone()[0]
        account_id1 = int(total_accts)+1
        cursor.execute('''INSERT INTO users(account_id, last_name, first_name, username, email)
                            VALUES(?,?,?,?,?)''', (account_id1, lastname1, firstname1, username1, email1))
        print("New user enrolled on truCOR platform.")
        db.commit()

        now = dt.now()
        today = str(now.isoformat())

        user = {
            "truCOR Response":{
                'Status': "New user enrolled on truCOR platform.",
                'Request Date': today
            },
            "Profile": {
                'ACCOUNT ID': account_id1,
                'LAST NAME': lastname1,
                'FIRST NAME': firstname1,
                'EMAIL': email1,
                'USERNAME': username1
            }
        }

    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()
    return jsonify({'user': user}), 201

@app.route('/api/v1.0/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    db = get_db()
    cursor = db.cursor()
    row = cursor.execute('''SELECT * FROM users WHERE account_id=?''', (user_id,))
    display_user = [{'Profile':{'ACCOUNT ID': user[0],
                                'LAST NAME': user[1],
                                'FIRST NAME': user[2],
                                'EMAIL': user[4],
                                'USERNAME': user[3]}} for user in row]
    return jsonify(display_user)

@app.route('/api/v1.0/users/<int:user_id>', methods=['PUT'])

@app.route('/api/v1.0/template', methods=['POST'])
def imprint_template():
    if not request.json or not 'USERNAME' in request.json:
            abort(400)
    db = get_db()
    cursor = db.cursor()
    username_in = request.json['USERNAME']
    cursor.execute('''SELECT account_id FROM users WHERE username=?''', (username_in,))
    account_ref = cursor.fetchone()[0]
    raw_data, sample_rate = sc.collect_EKG(device="EKG Monitor", duration=45)

    heart_template = et.set_template(raw_data, sample_rate)
    heart_template['created_at'] = dt.today()
    heart_template['associated_account'] = account_ref
    heart_template.to_sql("ekgtemplate", db, if_exists='append', index=False)
    db.commit()
    db.close()
    print('ECG template saved.')

    now = dt.now()
    today = str(now.isoformat())
    result = {
        "truCOR Response": {
            'Status': "Successful creation of EKG Template.",
            'Created': today
        }
    }
    return jsonify({'result': result}), 201

@app.route('/api/v1.0/identify', methods=['POST'])
def identify_sample():
    db = get_db()
    cursor = db.cursor()
    all_query = '''SELECT * FROM ekgtemplate'''
    all_data = pd.io.sql.read_sql(all_query, db)

    y_potato = all_data['associated_account']
    X_meat = all_data.drop(['created_at', 'associated_account'], axis=1)

    X_train = X_meat.as_matrix()
    y_train = y_potato.as_matrix()

    raw_data, sample_rate = sc.collect_EKG(device="EKG Monitor", duration=15)
    X_test = et.set_template(raw_data, sample_rate)

    #th_dataset = et.personaldata()
    #X_test = et.set_template(th_dataset, 228)

    account_ref = et.heart_identify(X_test, X_train, y_train)
    cursor.execute('''SELECT first_name, last_name, username FROM users WHERE account_id=?''', (int(account_ref),))
    row = cursor.fetchone()
    
    if row is not None:
        f_name_guess = row[0]
        l_name_guess = row[1]
        username_guess = row[2]
    else:
        f_name_guess = "Inconclusive"
        l_name_guess = "Result"

    now = dt.now()
    today = str(now.isoformat())

    result = {
        "truCOR Response": {
            'Status': f_name_guess + " " + l_name_guess + " has been identified.",
            'Last Updated': today,
            'Location': "truCOR - Bay Area Site."
        }
    }
    return jsonify({'result': result}), 201

@app.route('/api/v1.0/authn', methods=['POST'])
def heart_authn():
    if not request.json or not 'USERNAME' in request.json:
            abort(400)
    db = get_db()
    cursor = db.cursor()
    username_in = request.json['USERNAME']
    cursor.execute('''SELECT account_id FROM users WHERE username=?''', (username_in,))
    s = cursor.fetchone()[0]
    query = '''SELECT * FROM ekgtemplate WHERE associated_account=%s''' % s
    tagged_template_df = pd.io.sql.read_sql(query, db)
    
    not_query = '''SELECT * FROM ekgtemplate WHERE associated_account!=%s''' % s
    negative_template_df = pd.io.sql.read_sql(not_query,db)

    raw_data, sample_rate = sc.collect_EKG(device="EKG Monitor", duration=15)
    X_test = et.set_template(raw_data, sample_rate)

    #th_dataset = et.personaldata()
    #X_test = et.set_template(th_dataset, 228)

    truCOR_AuthN, match_pct = et.authenticate(X_test, tagged_template_df, negative_template_df)

    now = dt.now()
    today = str(now.isoformat())

    result = {
        "truCOR Response": {
            'User Authenticated': truCOR_AuthN,
            'Description': "EKG Sample was a " + str(match_pct*100) + " percent match.",
            'Last Updated': today
        }
    }
    return jsonify({'result': result}), 201


@app.route('/api/v1.0/reset', methods=['POST'])
def reset_template():
    if not request.json or not 'USERNAME' in request.json:
            abort(400)
    db = get_db()
    cursor = db.cursor()
    username_in = request.json['USERNAME']
    cursor.execute('''SELECT account_id FROM users WHERE username=?''', (username_in,))
    s = cursor.fetchone()[0]
    sql_delete = '''DELETE FROM ekgtemplate WHERE associated_account=%s''' % s
    cursor.execute(sql_delete)
    now = dt.now()
    today = str(now.isoformat())
    result = {
        "truCOR Response": {
            'Status': "EKG Template has been reset for " + username_in + ".",
            'Last Updated': today
            }
        }
    return jsonify({'result': result}), 201

@app.teardown_appcontext
def close_db(error):
    if hasattr(g, 'sqlite_db'):
        g.sqlite_db.close()