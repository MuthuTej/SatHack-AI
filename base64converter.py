import requests
import base64

# Encode the image
with open("image.png", "rb") as img:
    b64_img = base64.b64encode(img.read()).decode("utf-8")

# API URL
url = "http://127.0.0.1:8000/predict_base64"

# Payload
data = {"image": b64_img}

# Send POST request
response = requests.post(url, json=data)

print(response.status_code)
print(response.json())
