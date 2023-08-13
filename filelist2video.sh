#!/bin/sh

# https://trac.ffmpeg.org/wiki/Concatenate

printf "file '%s'\n" *.jpg > mylist.txt
ffmpeg -f concat -safe 0 -i mylist.txt -c copy output-`date +%Y-%m-%d_%H-%M-%S`.mp4
rm -f mylist.txt

