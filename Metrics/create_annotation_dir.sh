#!/bin/bash

# Check if an argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <directory_name>"
    exit 1
fi

# Define the main directory based on the argument
main_dir="$1"

# Check if the directory already exists
if [ -d "$main_dir" ]; then
    echo "Error: Directory '$main_dir' already exists. Aborting."
    exit 1
fi

# Create the main directory
mkdir -p "$main_dir"

# Create the five subdirectories inside the main directory
mkdir -p "$main_dir/1_hauptthema"
mkdir -p "$main_dir/2_nebenthema"
mkdir -p "$main_dir/3_kein_thema"
mkdir -p "$main_dir/4_unklar"
mkdir -p "$main_dir/corpus_full"

echo "Directory structure created successfully in '$main_dir'."
