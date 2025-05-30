# test_api.py
import requests

url = "http://localhost:8000/predict"

data = {
    "CPC": 1000.22,
    "category_id": "291000",
    "industry": "Marketing",
    "publisher": "2b764",
    "customer_id": "746",
    "market_id": "12733593",
    "conversions": 1
}

response = requests.post(url, json=data)
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
