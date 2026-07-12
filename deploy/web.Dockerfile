FROM nginx:alpine

COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html /usr/share/nginx/html/index.html
COPY snake/index.html /usr/share/nginx/html/snake.html
COPY wolf/index.html /usr/share/nginx/html/wolf.html