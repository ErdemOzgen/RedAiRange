#!/bin/bash

# Accept path as input
path=$1

# Check if the input is a file
if [ -f "$path" ]; then
    # Perform modelscan -p on the file
    modelscan -p "$path"
# Check if the input is a directory
elif [ -d "$path" ]; then
    # Perform modelscan -p on the saved_model.pb file in the directory
    modelscan -p "$path/saved_model.pb"
else
    echo "Invalid input. Please provide a valid file or directory path."
fi