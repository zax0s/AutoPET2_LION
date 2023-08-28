#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

#docker build --no-cache -t lion "$SCRIPTPATH"
docker build -t lion "$SCRIPTPATH"
