import base64
import json
import os
import urllib

import requests

if __name__ == "__main__":
    access_key = os.getenv("ACCESS_KEY")
    secret_key = os.getenv("SECRET_KEY")
    url = "to_be_replaced"

    # Encode the credentials using base64
    credentials = f"{access_key}:{secret_key}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    print("access_key:", access_key)
    print("secret_key:", secret_key)
    print("encoded credentials:", encoded_credentials)

    code = """
    #include <iostream>
    using namespace std;

    int main() {
        cout << "Bytedance! Leading the whole Universe! (Except for the $8,000,000 intern)" << std::endl;
        return 0;
    }
    """

    response = requests.post(
        urllib.parse.urljoin(url, "run_code"),
        headers={
            "Content-Type": "application/json",
            "X-Brain-Authorization": f"Basic {encoded_credentials}"
        },
        json={
            "code": code,
            "language": "cpp"
        },
        timeout=300,  # seconds
    )

    print(response.text)
    print(json.dumps(response.json(), indent=4))
