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

# ── RSA key resolution ──
# Priority: env var content (for Render/production) → file path (for local dev)
# On Render, set JWT_PRIVATE_KEY and JWT_PUBLIC_KEY as environment variables
# with the full PEM content (newlines as \n).

def _resolve_key(content_env, path_env, default_file):
    """Return path to PEM file. If content is in env var, write it to disk first."""
    content = os.getenv(content_env)
    if content:
        # PEM content passed directly as env var — write to temp file
        tmp_path = os.path.join(BASE_DIR, "secrets", default_file)
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        # Replace literal \n with actual newlines (Render env vars)
        content = content.replace("\\n", "\n")
        with open(tmp_path, "w") as f:
            f.write(content)
        return tmp_path
    return get_abs_path(path_env, default_file)

private_key = _resolve_key("JWT_PRIVATE_KEY", "JWT_PRIVATE_KEY_PATH", "private.pem")
public_key = _resolve_key("JWT_PUBLIC_KEY", "JWT_PUBLIC_KEY_PATH", "public.pem")
