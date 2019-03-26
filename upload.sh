#!/usr/bin/env bash
echo "uploading...."
path=`cd $(dirname $0);pwd -P`
echo the current path is:$path
for file in `ls`
do
    if [ "${file##*.}" = "py" ];then
	scp -P 3135 $file lijin@202.120.36.100:~/distilation
    fi
done
echo "Done!"
