FROM nginx:alpine

COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html /usr/share/nginx/html/
COPY favicon.svg favicon.png /usr/share/nginx/html/

# demo pages keep their repo paths, so local and deployed URLs coincide
COPY snake/index.html /usr/share/nginx/html/snake/index.html
COPY wolf/index.html  /usr/share/nginx/html/wolf/index.html
COPY life/index.html  /usr/share/nginx/html/life/index.html
COPY doom/index.html  /usr/share/nginx/html/doom/index.html
COPY kv/index.html    /usr/share/nginx/html/kv/index.html
COPY base64/index.html    /usr/share/nginx/html/base64/index.html
COPY www/index.html   /usr/share/nginx/html/www/index.html
# sw.js E' il server del sito compilato: senza, l'iframe di www/ resta vuoto.
# Va servito da /www/ perche' il suo scope (/www/neural/) sta sotto la sua path
COPY www/sw.js        /usr/share/nginx/html/www/sw.js

# legacy flat URLs, kept for old bookmarks
COPY snake/index.html /usr/share/nginx/html/snake.html
COPY wolf/index.html  /usr/share/nginx/html/wolf.html
COPY life/index.html  /usr/share/nginx/html/life.html
