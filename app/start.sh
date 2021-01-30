#! /usr/bin/env sh

python3 -V
python3 download_data_model.py
python3 database_init.py
exec gunicorn -c gunicorn_conf.py -k uvicorn.workers.UvicornWorker app:app
# exec uvicorn app:app --reload --host 0.0.0.0 --port 8000
