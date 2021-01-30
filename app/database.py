
from sqlalchemy import create_engine
from sqlalchemy import Column, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime as dt
from dotenv import load_dotenv

load_dotenv('.config')
POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'postgres')
POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')

db_string = "postgres://postgres:postgres@" + POSTGRES_HOST + ":" + POSTGRES_PORT + "/postgres"
print(db_string)
db = create_engine(db_string)

base = declarative_base()

class SearchHistory(base):
    __tablename__ = 'search_history'

    search_time = Column(DateTime, primary_key=True)
    search_text = Column(String)

Session = sessionmaker(db)
session = Session()

# Create
def add_search_history(text):
  search_time = dt.now()
  item = SearchHistory(search_time=search_time, search_text=text)
  # item = SearchHistory(search_text=text)
  session.add(item)
  session.commit()

# Read
def read_search_history():
  history = session.query(SearchHistory)
  print(type(history))
  results = []
  for h in history:
    results.append(h.search_text)
  return results
