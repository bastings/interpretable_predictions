#!/usr/bin/env bash

echo "-- Beer data --"
mkdir -p data/beer
cd data/beer

echo "Downloading test data"
wget wget -q --show-progress http://people.csail.mit.edu/taolei/beer/annotations.json

echo "Downloading word embeddings"
wget wget -q --show-progress http://people.csail.mit.edu/taolei/beer/review+wiki.filtered.200.txt.gz

echo "Downloading combined train/dev data"
wget wget -q --show-progress http://people.csail.mit.edu/taolei/beer/reviews.260k.heldout.txt.gz
wget wget -q --show-progress http://people.csail.mit.edu/taolei/beer/reviews.260k.train.txt.gz

echo "Downloading aspect 0 train/dev data"
wget wget -q --show-progress http://people.csail.mit.edu/taolei/beer/reviews.aspect0.heldout.txt.gz
wget wget -q --show-progress http://people.csail.mit.edu/taolei/beer/reviews.aspect0.train.txt.gz

echo "Downloading aspect 1 train/dev data"
wget wget -q --show-progress http://people.csail.mit.edu/taolei/beer/reviews.aspect1.heldout.txt.gz
wget wget -q --show-progress http://people.csail.mit.edu/taolei/beer/reviews.aspect1.train.txt.gz

echo "Downloading aspect 2 train/dev data"
wget wget -q --show-progress http://people.csail.mit.edu/taolei/beer/reviews.aspect2.heldout.txt.gz
wget wget -q --show-progress http://people.csail.mit.edu/taolei/beer/reviews.aspect2.train.txt.gz

cd ..
cd ..

echo "-- SST --"
mkdir -p data/sst
cd data/sst

echo "Downloading SST data"
wget https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip
unzip -j trainDevTestTrees_PTB.zip 

echo "Downloading filtered word embeddings"
wget https://gist.github.com/bastings/b094de2813da58056a05e8e7950d4ad1/raw/3fbd3976199c2b88de2ae62afc0ecc6f15e6f7ce/glove.840B.300d.sst.txt

cd ..
cd ..

echo "-- SNLI --"
echo "Download SNLI word list"
mkdir -p data/snli
cd data/snli
wget https://gist.github.com/bastings/1c8f40ff7c9a5f3eddc259fea319c332/raw/b205be7f42e99de9047118019c72f01d7880b158/glove.840B.300d.snli.txt

