import urllib.request
import os
import json
from dotenv import load_dotenv

load_dotenv()
key = os.getenv('FOURSQUARE_API_KEY').strip('"').strip("'")
print(f"Original Key: {key}")

key_fsq3 = "fsq3" + key
print(f"Testing Key: {key_fsq3}")

req = urllib.request.Request('https://api.foursquare.com/v3/places/search?near=Delhi,India&query=Temple&limit=1', headers={
    'Authorization': key_fsq3,
    'accept': 'application/json'
})

try:
    r = urllib.request.urlopen(req)
    data = json.loads(r.read().decode())
    print("Success!", data['results'][0]['name'])
except Exception as e:
    print("Failed:", getattr(e, 'code', 'Unknown'), getattr(e, 'read', lambda: b'')().decode())
