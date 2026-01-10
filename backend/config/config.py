from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

database_url = os.getenv("DATABASE_URL")

private_key = os.path.join(BASE_DIR, ".keys", "private.pem")
public_key = os.path.join(BASE_DIR, ".keys", "public.pem")
