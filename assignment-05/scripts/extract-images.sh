#!/bin/bash

IMAGES_DIR=images

[ -d "$IMAGES_DIR" ] && [ "$(ls -A $IMAGES_DIR)" ] && echo "WARNING: Images Already Extracted" && exit 0

mkdir $IMAGES_DIR

cd videos

for vid in *; do
	ffmpeg -i "$vid" -r 1 -q:v 2 -f image2 "$vid"-image-3%d.jpg
done

mv *.jpg ../$IMAGES_DIR
