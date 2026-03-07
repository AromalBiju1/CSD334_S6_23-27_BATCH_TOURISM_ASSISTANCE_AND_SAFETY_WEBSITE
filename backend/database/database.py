import os
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import create_engine

from config.config import database_url

# Render provides postgres:// but SQLAlchemy needs postgresql://
db_url = database_url

if not db_url:
    # Build steps on platforms like Render may import this file without env vars
    db_url = "sqlite:///./test.db"
elif db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

engine = create_engine(
    db_url,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=300,
)
sessionLocal = sessionmaker(autoflush=False,autocommit=False,bind=engine)
Base = declarative_base()


def get_db():
    db = sessionLocal()
    try:
        yield db
    finally:
        db.close()    