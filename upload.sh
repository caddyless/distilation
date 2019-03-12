#!/usr/bin/env bash

echo "uploading...."
path='/home/lijin/PycharmProjects/distilation'
data="data"
upload="upload.sh"
filelist=$(ls $path)
for file in $filelist
do
    if [[ "$file" != "$data" && "$file" != "$upload" ]]; then
        scp -P 3135 $file lijin@202.120.36.100:~/distilation
    fi
done
