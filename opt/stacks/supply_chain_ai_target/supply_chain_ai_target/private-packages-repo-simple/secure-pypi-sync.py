#!/usr/bin/env python3

import requests
from bs4 import BeautifulSoup
import subprocess
import os
import shutil
import logging

def setup_logging():
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    
    file_handler = logging.FileHandler('pypi_packages.log')
    file_handler.setFormatter(logging.Formatter(log_format))
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    
    logging.getLogger().addHandler(file_handler)
    #logging.getLogger().addHandler(console_handler)

def download_package_list():
    logging.info("Starting download of package list from PyPI.")
    try:
        # Fetch the HTML content from PyPI Simple API
        response = requests.get('https://pypi.org/simple/')
        response.raise_for_status()
    except requests.HTTPError as err:
        logging.error(f"HTTP Error occurred: {err}")
        return
    except Exception as err:
        logging.error(f"An error occurred: {err}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')

    # Open a file in write mode to store package names
    with open('pypi_packages.txt', 'w') as f:
        links_list = soup.find_all('a')
        logging.info(f"List contains {len(links_list)} packages.")
        i= 0;
        for anchor in links_list:
            package_name = anchor.string
            f.write(f"{package_name}\n")
            i = i + 1
    logging.info(f"Finished downloading package list from PyPI. {i} files added to pypi_packages.txt'")

def fetch_pypi_package_list(limit=5):
    with open('pypi_packages.txt', 'r') as f:
        packages = [line.strip() for line in f][:limit]
    return packages

def upload_to_private_pypi(file_path):
    private_pypi_dir = os.path.expanduser("~/packages")
    
    if not os.path.exists(private_pypi_dir):
        os.makedirs(private_pypi_dir)

    shutil.copy(file_path, private_pypi_dir)

def main():
    setup_logging()
    download_package_list()
    
    packages = fetch_pypi_package_list(limit=5)

    temp_directory = "./tmp/pypi"
    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)
    
    for package in packages:
        logging.info(f"Attempting to download package: {package}")
        
        try:
            subprocess.run(["pip", "download", "--no-cache-dir", package, "-d", temp_directory], check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to download package {package}: {e}")
            continue

        package_files = [f for f in os.listdir(temp_directory) if package in f]

        for package_file in package_files:
            package_path = os.path.join(temp_directory, package_file)
            logging.info(f"Scanning {package_path} for vulnerabilities.")
            result = subprocess.run(["trivy", "fs", "--exit-code", "1", "--severity", "CRITICAL,HIGH", package_path], capture_output=True, text=True)
            if result.returncode == 0:
                upload_to_private_pypi(package_path)
                logging.info(f"Uploaded {package_path} to private PyPI.")
            else:
                logging.warning(f"Skipping {package_path} due to vulnerabilities.")
        
   # shutil.rmtree(temp_directory)

if __name__ == "__main__":
    main()
