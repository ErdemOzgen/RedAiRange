#!/bin/bash

# Function to scan a single notebook
scan_notebook() {
    local notebook="$1"
    local script_name="${code_dir}/${notebook%.ipynb}.py"
    jupyter nbconvert --to script --output-dir="$code_dir" "$notebook"
    bandit "$script_name"
    [[ -z "$keep_flag" ]] && rm -f "$script_name"
}

# Check for -r and -k parameters
recursive_flag=""
keep_flag=""
while [[ "$1" =~ ^- && ! "$1" == "--" ]]; do
    case $1 in
        -r )
            recursive_flag="-r"
            ;;
        -k )
            keep_flag="1"
            ;;
    esac
    shift
done
if [[ "$1" == '--' ]]; then shift; fi

# Check if directory is provided
if [[ -z "$1" ]]; then
    echo "Usage: $0 [-r] [-k] directory"
    exit 1
fi

directory="$1"

# Create the code-YYYYMMDD-HHMMSS subdirectory
timestamp=$(date +"%Y%m%d-%H%M%S")
code_dir="$directory/code-$timestamp"
mkdir -p "$code_dir"

# Find all notebooks, generate Python files in the code-YYYYMMDD-HHMMSS subdirectory, and scan them
if [[ -n "$recursive_flag" ]]; then
    find "$directory" -type f -name "*.ipynb" -print0 | while IFS= read -r -d '' notebook; do
        scan_notebook "$notebook"
    done
else
    for notebook in "$directory"/*.ipynb; do
        [[ -f "$notebook" ]] && scan_notebook "$notebook"
    done
fi

# Delete the code-YYYYMMDD-HHMMSS subdirectory if -k is not specified
[[ -z "$keep_flag" ]] && rm -rf "$code_dir"
