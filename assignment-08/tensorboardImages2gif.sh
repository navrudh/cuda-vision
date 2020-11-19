#!/bin/bash

status_code=200
i=0

if [ ! -f "/tmp/tb_img-0.jpg" ]; then
  while [[ "$status_code" -eq 200 ]] ; do
    status_code=$(curl --write-out %{http_code} --output "/tmp/tb_img-$i.jpg" --silent "http://localhost:6006/data/plugin/images/individualImage?ts=1598042496.381562&run=version_5&tag=generated_images&sample=0&index=$i")
    ((i=i+1))
  done
fi

convert -delay 20 -loop 0 "/tmp/tb_img-*.jpg" gan-animation.gif