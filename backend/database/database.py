import os
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine

from config.config import database_url

engine = create_engine(database_url)
sessionLocal = sessionmaker(autoflush=False,autocommit=False,bind=engine)
Base = declarative_base()


def get_db():
    db = sessionLocal()
    try:
        yield db
    finally:
        db.close()    