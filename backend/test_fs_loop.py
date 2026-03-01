import urllib.request
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()
key = os.getenv('FOURSQUARE_API_KEY').strip('"').strip("'")

cities = ["Mumbai", "Delhi", "Alapuzha", "Agra", "Adilabad"]

for city in cities:
    req = urllib.request.Request(f'https://api.foursquare.com/v3/places/search?near={city},India&query=Temple&limit=1', headers={
        'Authorization': key,
        'accept': 'application/json'
    })
    try:
        r = urllib.request.urlopen(req)
        data = json.loads(r.read().decode())
        print(f"Success {city}:", data.get('results', [{}])[0].get('name'))
    except Exception as e:
        print(f"Failed {city}:", e.code, getattr(e, 'read', lambda: b'')().decode())
    time.sleep(1)
