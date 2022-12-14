#!/usr/local/bin/bash

for file in /Users/axelahdritz/Desktop/thesis/batch1/*; do
    python3 speech-to-text.py "--audio_file=$file" &
done
