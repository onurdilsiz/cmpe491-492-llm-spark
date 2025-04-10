import requests

# Base URL for the Spark History Server or running Spark application
base_url = "http://10.164.0.58:18080/api/v1"  # Replace with your Spark History Server URL
# For a running application, use: base_url = "http://<driver-host>:4040/api/v1"

# Endpoint to get all applications
endpoint = f"{base_url}/applications"

# Optional query parameters
params = {
    "status": "completed",  # Filter by status: "completed" or "running"

    "limit": 10  # Limit the number of applications returned (optional)
}

# Make the GET request
response = requests.get(endpoint, params=params)

# Check if the request was successful
if response.status_code == 200:
    applications = response.json()
    print("Applications:")
    for app in applications:
        print(f"ID: {app['id']}, Name: {app['name']}, Start Time: {app['attempts'][0]['startTime']}")
else:
    print(f"Failed to retrieve applications. Status code: {response.status_code}")
    print(f"Response: {response.text}")