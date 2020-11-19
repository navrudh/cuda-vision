#!/bin/bash

for img in *; do
	convert -gravity East -crop "${1}x${1}" "${img}" "${img}_cropped_%d.${img##*.}"
done

echo "Done!"
