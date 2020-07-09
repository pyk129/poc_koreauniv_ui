#!/bin/bash

APP=estate_ui

docker stop $APP

docker rm $APP
:
docker run -d -p 8801:8080 --name=$APP estate_ui
 
