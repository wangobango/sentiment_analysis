#!/bin/bash
FILEID=0B0BvGSZqvqJPbUU5NDFjbkV6a0U
FILENAME=dranziera_dataset.zip
if [ ! -f "$FILENAME" ]; then
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$FILEID -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$FILEID" -O $FILENAME && rm -rf /tmp/cookies.txt
fi
if [ ! -d "data" ]; then
unzip $FILENAME -d data
fi
