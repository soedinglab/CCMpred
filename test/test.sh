#!/bin/bash
source_dir=$PWD
temp_dir=`mktemp -d`

cd $temp_dir
cmake $source_dir -DWITH_CUDA=off
make
bin/ccmpred $source_dir/example/1atzA.aln 1atzA.mat
[[ $(wc -l < 1atzA.mat) == 75 ]]
