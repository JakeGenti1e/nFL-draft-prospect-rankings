import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import os
import requests
BASE_URL = "https://api.collegefootballdata.com"

API_KEY = os.environ.get("CFBD_API_KEY")

def get_playerInfo(): 
        response = requests.get(...)
        data = response.json()
        #save_raw(data, f"player_info_{year}.json")