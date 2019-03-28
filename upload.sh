#!/usr/bin/env bash
echo "uploading...."
path=`cd $(dirname $0);pwd -P`
echo the current path is:$path
scp -P 3135 *.py lijin@202.120.36.100:~/distilation
echo "Done!"
