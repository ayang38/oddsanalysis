import requests
import secret
import pprint
import json
from datetime import datetime
import os
import time

API_KEY = secret.ODDS_API_KEY
SPORT = 'upcoming'
REGIONS = 'us'
MARKETS = 'h2h,spreads,totals' 
ODDS_FORMAT = 'decimal'
DATAFOLDER = 'DATAFOLDER'
url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds'

params = {
'apiKey' : API_KEY,
'regions':REGIONS, 
'markets': MARKETS,
'oddsFormat': ODDS_FORMAT
}
i = 0
while True:
    try:
        os.makedirs(DATAFOLDER, exist_ok=True)
        response = requests.get(url, params=params)
        response.raise_for_status() #checks for errors
        data = response.json()
        filename = f"snapshot{datetime.now().strftime('%M%d%Y_%H%M%S')}.json"
        path = os.path.join(DATAFOLDER, filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Data saved to {filename}")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
    i += 1
    sleeping_time = 900
    print(f'It is currently {datetime.now()} Sleeping for {sleeping_time} seconds. Currently have {i} snapshots.')
    time.sleep(sleeping_time)
