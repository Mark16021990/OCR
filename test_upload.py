import requests

url = 'http://localhost:5000/upload'
file_path = 'test_passport.jpg'  # Убедитесь, что файл существует

with open(file_path, 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")