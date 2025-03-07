#!/usr/bin/env python3
import os
import re
import sys

# Check if a path is provided
if len(sys.argv) < 2:
    print(f"Usage: python {sys.argv[0]} /path/to/packages/folder")
    sys.exit(1)

# Get the path from the command-line argument
folder_path = sys.argv[1]

# Navigate to the folder containing the packages
try:
    os.chdir(folder_path)
except FileNotFoundError:
    print("Directory does not exist")
    sys.exit(1)

# List all files in the directory
files = os.listdir()

# Initialize an empty list to store package names and versions
requirements = []

# Regular expression to match package name and version
pattern = re.compile(r"([a-zA-Z0-9\-_]+)-([0-9\.]+)")

for file in files:
    match = pattern.search(file)
    if match:
        package_name = match.group(1)
        package_version = match.group(2)
        requirements.append(f"{package_name}=={package_version}")

# Write the requirements to a file
with open("requirements.txt", "w") as f:
    for requirement in requirements:
        f.write(f"{requirement}\n")

print(f"Generated requirements.txt in {folder_path}")
