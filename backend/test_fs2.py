import urllib.request
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv('FOURSQUARE_API_KEY').strip('"').strip("'")
print(f"Key: {key}")

req = urllib.request.Request('https://api.foursquare.com/v3/places/search?near=Alapuzha,Kerala,India&query=Park&limit=1', headers={
    'Authorization': key,
    'accept': 'application/json'
})

try:
    r = urllib.request.urlopen(req)
    print("Success:", r.read().decode())
except Exception as e:
    print(e.code, getattr(e, 'read', lambda: b'')().decode())
