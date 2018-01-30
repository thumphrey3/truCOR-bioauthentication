#!/bin/bash
export FLASK_APP=./trucor/index.py
source $(pipenv --venv)/bin/activate
flask initdb
flask run -h 0.0.0.0