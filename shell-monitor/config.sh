#!/bin/bash

#deploy_dir=$HOME/deploy/inforec/img_search_srv/9191
#deploy_dir=$HOME/cbir/framework_v0.2
deploy_dir=$HOME/llh/classify_quanquan

GLIBC_DIR=$HOME/llh/software/glibc-2.23/lib
MYPYTHON=$HOME/llh/software/anaconda3/bin/python
PREFIX_MYPYTHON="$GLIBC_DIR/ld-2.23.so --library-path $GLIBC_DIR:/lib:/lib/x86_64-linux-gnu:/usr/lib:/usr/lib/x86_64-linux-gnu "


#echo '##################'
#echo $PORT
#echo $deploy_dir
#echo $GLIBC_DIR
#echo $MYPYTHON
#echo $PREFIX_MYPYTHON
#echo '##################'
