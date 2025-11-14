import firebase_admin
from firebase_admin import credentials, db
import random
import time

# Path to your Firebase service account key JSON file
cred = credentials.Certificate("serviceAccountKey.json")

# Initialize the app with a database URL
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://sathack-7d9d8-default-rtdb.firebaseio.com/'
})

# Reference to the database location where you want to write data
ref = db.reference('random_values')  # e.g., "random_values" node

# Define your range
min_value = 10
max_value = 100

# Continuous loop to write random values
while True:
    random_value = random.randint(min_value, max_value)
    ref.push(random_value)  # Use push() to create a unique key for each value
    print(f"Written value: {random_value}")
    time.sleep(1)  # wait 1 second before writing the next value
