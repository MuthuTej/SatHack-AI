import os
import json
import firebase_admin
from firebase_admin import credentials, db

# Check if running in Render
if os.getenv("RENDER"):
    # Get Firebase credentials from environment variable
    firebase_creds = os.getenv("FIREBASE_CREDENTIALS")
    if not firebase_creds:
        raise ValueError("FIREBASE_CREDENTIALS not found in environment variables")

    # Convert string to dict
    cred_info = json.loads(firebase_creds)
else:
    # Local mode: use the serviceAccountKey.json file
    with open("serviceAccountKey.json") as f:
        cred_info = json.load(f)

# Initialize Firebase
cred = credentials.Certificate(cred_info)
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://sathack-7d9d8-default-rtdb.firebaseio.com/"
})

database = db
