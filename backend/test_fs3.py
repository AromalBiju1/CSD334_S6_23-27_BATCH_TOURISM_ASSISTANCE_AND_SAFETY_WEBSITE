import urllib.request
import os
import json
from dotenv import load_dotenv

load_dotenv()
key = os.getenv('FOURSQUARE_API_KEY').strip('"').strip("'")
print(f"Key: {key}")

try:
    print("Testing Foursquare V3 API...")
    req = urllib.request.Request('https://api.foursquare.com/v3/places/search?near=Mumbai,India&query=Temple&limit=1', headers={
        'Authorization': key,
        'accept': 'application/json'
    })
    r = urllib.request.urlopen(req)
    data = json.loads(r.read().decode())
    print("V3 Success:", data.get('results', [{}])[0].get('name'))
except Exception as e:
    print("V3 failed:", e.code, getattr(e, 'read', lambda: b'')().decode())

try:
    print("\nTesting Foursquare V2 API...")
    url = f"https://api.foursquare.com/v2/venues/search?near=Mumbai,India&query=Temple&oauth_token={key}&v=20231010&limit=1"
    req = urllib.request.Request(url)
    r = urllib.request.urlopen(req)
    data = json.loads(r.read().decode())
    print("V2 Success:", data.get('response', {}).get('venues', [{}])[0].get('name'))
except Exception as e:
    print("V2 failed:", e.code, getattr(e, 'read', lambda: b'')().decode())
