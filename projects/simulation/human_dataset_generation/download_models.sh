wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1poXfXjg8mwG_7CJE58yihoWzLbrIyi41' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1poXfXjg8mwG_7CJE58yihoWzLbrIyi41" -O human_models.tar.gz && rm -rf /tmp/cookies.txt

tar -xzvf human_models.tar.gz

rm human_models.tar.gz
