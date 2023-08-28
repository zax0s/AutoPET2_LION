#!/usr/bin/env bash

./build.sh

docker save lion | gzip -c > LION.tar.gz
