import urllib.request
import urllib.error
import urllib.parse
import os
from dotenv import load_dotenv

load_dotenv()
key = os.getenv('FOURSQUARE_API_KEY').strip('"').strip("'")
print(f"Key: {key}")

# Try v3
try:
    req = urllib.request.Request('https://api.foursquare.com/v3/places/search?near=Delhi,India&query=Temple&limit=1', headers={
        'Authorization': key,
        'accept': 'application/json'
    })
    r = urllib.request.urlopen(req)
    print("V3 success!")
    print(r.read().decode())
except urllib.error.HTTPError as e:
    print(f"V3 failed: {e.code} {e.read().decode()}")

# Try v2 with oauth_token
try:
    url = f"https://api.foursquare.com/v2/venues/search?near=Delhi,India&query=Temple&oauth_token={key}&v=20231010&limit=1"
    req = urllib.request.Request(url)
    r = urllib.request.urlopen(req)
    print("V2 success!")
    print(r.read().decode())
except urllib.error.HTTPError as e:
    print(f"V2 failed: {e.code} {e.read().decode()}")

