from google.cloud import secretmanager

if __name__ == "__main__":
    project_id = "84009481424"
    secret_id = "GEMINI_API_KEY"
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    result = response.payload.data.decode("UTF-8")
    print(f"Secret value: {result}")