#!/bin/bash

git pull

docker build --no-cache -t estate_ui .
