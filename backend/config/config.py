from dotenv import load_dotenv
import os

load_dotenv()

database_url = os.getenv("DATABASE_URL")
private_key = os.getenv("JWT_PRIVATE_KEY_PATH")
public_key = os.getenv("JWT_PUBLIC_KEY_PATH")


