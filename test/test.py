# apify_linkedin_fetch.py
import os
import json
import requests
from dotenv import load_dotenv

def main():
    load_dotenv()

    APIFY_TOKEN = os.getenv("APIFY_TOKEN")
    ACTOR_ID = os.getenv("ACTOR_ID")
    LINKEDIN_URL = "https://www.linkedin.com/in/jay-goenka-b797851b2/"

    if not APIFY_TOKEN:
        raise SystemExit("Set APIFY_TOKEN env var")

    url = f"https://api.apify.com/v2/acts/{ACTOR_ID}/run-sync-get-dataset-items"
    params = {"token": APIFY_TOKEN, "format": "json"}
    payload = {
        "profileUrls": [LINKEDIN_URL],
        "maxConcurrency": 1
    }

    r = requests.post(url, params=params, json=payload, timeout=300)
    r.raise_for_status()
    items = r.json()  # usually a list of 1 item per profile
    if not items:
        print("No data returned.")
        return

    # Print the raw profile JSON (you can map fields after you see the shape)
    print(json.dumps(items[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
