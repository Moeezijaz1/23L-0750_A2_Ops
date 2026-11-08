#!/usr/bin/env bash
# Task 6.1 - take a screenshot after each command.
docker pull nginx:alpine
docker run -d --name web1 -p 8080:80 nginx:alpine              # map host 8080 -> container 80
curl -I http://localhost:8080
mkdir -p site && echo "<h1>hello from a volume</h1>" > site/index.html
docker run -d --name web2 -p 8081:80 -v "$PWD/site:/usr/share/nginx/html:ro" nginx:alpine
curl http://localhost:8081                                      # shows the mounted file
docker logs web1
docker exec -it web1 sh                                         # (type: ls /etc/nginx; exit)
docker ps -a ; docker images
docker stop web1 web2 && docker rm web1 web2 && docker rmi nginx:alpine
