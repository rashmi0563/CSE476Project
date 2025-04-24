import requests

url = "http://localhost:8000/generate"

print("==== CSE476 - GG Team - API TEST ====")

while True:
    prompt = input("Prompt (Enter Exit to exit): ")
    print()

    if prompt.lower() == "exit":
        print("Bye!")
        break

    payload = {'prompt' : "Answer concisely: " + prompt}

    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("Response :", response.json()["response"])
            print("=" * 30)
        else:
            print(f"Error occur : {response.status_code} - {response.text}")
            print("=" * 30)
    except Exception as e:
        print(f"Failed to connect to server: {e}")
        print("=" * 30)
