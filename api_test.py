import requests

API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"

# Paste your NEW token here
headers = {"Authorization": "Bearer YOUR_TOKEN"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()

    response = requests.post(API_URL, headers=headers, data=data)

    print("Status Code:", response.status_code)

    try:
        return response.json()
    except:
        return response.text

result = query("test.jpg")

print("\nAPI Prediction:")
print(result)
