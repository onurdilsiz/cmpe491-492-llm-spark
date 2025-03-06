import requests
from google.oauth2 import service_account
from google.auth.transport.requests import Request
import google.auth
import os

def get_access_token_method1():
    """Get access token using google.auth.default()"""
    credentials, project = google.auth.default(
        scopes=['https://www.googleapis.com/auth/cloud-platform']
    )
    credentials.refresh(Request())
    return credentials.token



def download_spark_history():
    # Get the access token using either method
    access_token = get_access_token_method1()  
    
    # URL for the Spark History Server
    url = "https://pqbmrig4hndx5dwgt43tb3pq7a-dot-europe-west4.dataproc.googleusercontent.com/gateway/default/sparkhistory/history/app-20250219140302-0000/jobs/?user.name=anonymous"
    
    # Set up the headers with Proxy-Authorization
    headers = {
        "Proxy-Authorization": f"Bearer {access_token}"
    }
    
    # Make the request
    response = requests.get(url, headers=headers)
    
    # Check if request was successful
    response.raise_for_status()
    
    # Save the response to jobs.html
    with open("jobs.html", "w", encoding="utf-8") as f:
        f.write(response.text)
    
    print("Successfully downloaded jobs.html")
    return response.text

# Run the function
if __name__ == "__main__":
    try:
        download_spark_history()
    except Exception as e:
        print(f"Error: {e}")