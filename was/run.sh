#!/bin/bash

APP=estate_model

docker stop $APP

docker rm $APP
:
docker run -d -p 8802:5001 --name=$APP estate_model


