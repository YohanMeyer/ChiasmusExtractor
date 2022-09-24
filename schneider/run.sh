#!/bin/bash
GREEN="\033[0;32m"
NC="\033[0m"

play="schiller-wilhelm-tell"

# check if fasttext model is there
# if not, then download it
mkdir -p fasttext_models
[[ ! -f fasttext_models/wiki.en.bin ]] && \
    echo -e "${GREEN}### downloading English fasttext model from fasttext.cc${NC}" && \
    cd fasttext_models && \
    wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip && \
    unzip wiki.en.zip && cd .. && \
    echo -e "${GREEN}### done downloading English fasttext model${NC}"


echo -e "${GREEN}### download spacy model if not present${NC}"
# run only once
# python3 -m spacy download de_core_news_lg
# python3 -m spacy download en_core_web_lg
echo -e "${GREEN}### done downloading spacy model${NC}"

# finally run the example
echo -e "${GREEN}### running the experiment${NC}"
mkdir -p processed
mkdir -p candidates
python3 src/chiasmus_example.py ${play}
echo -e "${GREEN}### done running the experiment${NC}"
