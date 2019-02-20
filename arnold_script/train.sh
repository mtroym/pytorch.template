#!/bin/bash
THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $THIS_DIR # To run on Arnold
echo "\$THIS_DIR = ${THIS_DIR}
"
echo "copying data to the lab cluster..."
mkdir -pv /opt/tiger/data/segTHOR/train/
cp /mnt/cephfs/lab/tony/datasets/segTHOR/train/segTHOR.tar /opt/tiger/data/segTHOR/train/
echo "modified. -> /mnt/cephfs/lab/tony/datasets/cnnface.tar \$THIS_DIR"
echo "
decompressing data ..."
#echo "modified. -> tar -xf cnnface.tar"
cd /opt/tiger/data/
tar -xf segTHOR.tar
cd $THIS_DIR
cd ..
echo "envs:
========="
echo "\$GEN=${GEN}, \$MODELS=${MODELS}, \$WWW=${WWW}"
echo "================"
echo "arguements:
---------"
echo "${@}"
echo "----------"
exec $@
#python3 main.py --data ../data/ --gen ${GEN} --www ${WWW} --resume ${MODELS} --batchSize 4 --debug F --epochNum 0 --netType deeplab --dataset segTHOR --metrics '[]'