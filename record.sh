#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <filename.wav>"
  exit 1
fi

FILENAME="$1"
arecord -D hw:1,0 -f S16_LE -r 16000 -c2 "$FILENAME"
