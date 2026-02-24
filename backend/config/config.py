from dotenv import load_dotenv
import os

load_dotenv()

database_url = os.getenv("DATABASE_URL")

# Get the directory of the current file (config.py)
# Move up one level to the 'backend' folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_abs_path(env_key, default_filename):
    path = os.getenv(env_key)
    if not path or ":USERPROFILE" in path:
        # Fallback to a clean absolute path if the env is missing or broken
        return os.path.join(BASE_DIR, "secrets", default_filename)
    # If the path starts with ./, resolve it relative to BASE_DIR
    if path.startswith("./"):
        return os.path.join(BASE_DIR, path[2:])
    return path

private_key = get_abs_path("JWT_PRIVATE_KEY_PATH", "private.pem")
public_key = get_abs_path("JWT_PUBLIC_KEY_PATH", "public.pem")
