#!/bin/bash

vids=(
    "https://www.youtube.com/watch?v=JdP0AG0FqwY"
    "https://www.youtube.com/watch?v=6ldHWWHfeBc&t=5s"
    "https://www.youtube.com/watch?v=WJKc56uUuF8"
    "https://www.youtube.com/watch?v=RG205OwGdSg"
    "https://www.youtube.com/watch?v=yVdB_0ry53o"
    "https://www.youtube.com/watch?v=TWNvSHpMrSM"
    "https://www.youtube.com/watch?v=UsmBD2_3FH8"
    "https://www.youtube.com/watch?v=WGKo_6IkFBY"
    "https://www.youtube.com/watch?v=G6xE7uWt6Fo&t=1s"
    "https://www.youtube.com/watch?v=G9llFqAwI-8&t=17s"
    "https://www.youtube.com/watch?v=t8Ni5cB9FCc&t=21s"
    "https://www.youtube.com/watch?v=CLkcznptenE"
    "https://www.youtube.com/watch?v=-FQOZTsEA1Y"
    "https://www.youtube.com/watch?v=hkZAMO0y0Hs"
    "https://www.youtube.com/watch?v=9saVpA3wIbU"
    "https://www.youtube.com/watch?v=RC7ZNXclWWY"
)

mkdir -p videos
cd videos

for link in "${vids[@]}"; do
	youtube-dl -f "$1" "$link"
done

