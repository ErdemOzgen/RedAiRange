import requests
import argparse

# Define the URL of the API endpoint
url = "http://localhost:5000/predict"

def send_request(image_path):
    # Open the image file in binary mode
    with open(image_path, "rb") as image_file:
        # Define the files for the request
        files = {"file": image_file}

        # Make the POST request and store the response
        response = requests.post(url, files=files)

    # return  the JSON response
    return response.json()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Send a POST request to the Flask CIFAR-10 Inference Service with an image.')
    parser.add_argument('image_path', type=str, help='Path to the image to be sent.')

    args = parser.parse_args()

    if args.image_path:
        result = send_request(args.image_path)
        print (result)
    else:
        print("Please provide an image path. Use -h for help.")