#!/usr/local/bin/bash

NEW="/Users/axelahdritz/Desktop/thesis/audio_processed/"

for file in /Users/axelahdritz/Desktop/thesis/audio_files/*; do
    name=${file:46:$((${#file} - 46 - 0))}
    echo "$name"
    sox "$file" -b 16 "$NEW$name"
done

