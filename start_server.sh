#!/usr/bin/env bash

nodemon --exec 'FLASK_APP=server/server.py flask run -p 6000' --watch 'server/**.py' -e py