import requests

# Test the health endpoint
response = requests.get("http://localhost:8000/health")
print(response.json())

# Make a prediction
data = {
    "Title": 1,
    "Bathroom": 2.0,
    "Carpet Area": 1000.0,
    "location": 1,
    "Transaction": 1,
    "Furnishing": 1,
    "Balcony": 1.0,
    "facing": 1,
    "Price (in rupees)": 5000.0,
    "Status": 1,
    "Society": 1,
    "Floor": 1
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
