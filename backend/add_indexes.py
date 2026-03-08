import sys
import os
from sqlalchemy import text
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from database.database import engine

def create_indexes():
    with engine.connect() as conn:
        print("Adding index on rating...")
        # Committing transaction so we aren't in a lingering one
        conn.execute(text("COMMIT"))
        
        try:
            # PostgreSQL syntax
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_attractions_rating ON attractions (rating DESC NULLS LAST);"))
            print("Index on rating created.")
        except Exception as e:
            print(f"Error creating rating index: {e}")

        try:
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_attractions_category ON attractions (category);"))
            print("Index on category created.")
        except Exception as e:
            print(f"Error creating category index: {e}")

if __name__ == "__main__":
    create_indexes()
