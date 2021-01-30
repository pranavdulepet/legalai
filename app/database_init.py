
from sqlalchemy import create_engine
from sqlalchemy import Column, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv('.config')
POSTGRES_HOST = os.getenv('POSTGRES_HOST')
POSTGRES_PORT = os.getenv('POSTGRES_PORT')

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

# TODO: delete in prod
if db.has_table('search_history'):
  print(f"dropping table - search_history")
  SearchHistory.__table__.drop(db)

base.metadata.create_all(db)

session.close()
