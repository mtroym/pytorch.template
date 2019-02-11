#!/bin/bash
THIS_DIR="$( cd "$( dirname "$0" )" && pwd )"
cd $THIS_DIR # To run on Arnold
echo "\$THIS_DIR = ${THIS_DIR}
"
echo "copying data to the lab cluster..."
cp /mnt/cephfs/lab/tony/segTHOR.tar ../../
echo "modified. -> /mnt/cephfs/lab/tony/cnnface.tar \$THIS_DIR"
echo "
decompressing data ..."
#echo "modified. -> tar -xf cnnface.tar"
tar -xf segTHOR.tar
echo "done decompressing!
"
echo "arguements:
---------"
echo "${@}"
echo "----------"
exec $@
# Do like this.
# python3 main.py --data ./data/ --gen /mnt/cephfs/lab/tony/gen/  \
#                                --resume ./models/               \
#                                --www /mnt/cephfs/lab/tony/www/  \
#                                --epochNum 0 --batchSize 64      \

# & mv ../models/* /mnt/cephfs/lab/tony/models/
#python main.py --data ../data/ \
#               --gen ../gen/ \
#               --batchSize 1 --debug True \
#               --epochNum 0 --netType  deeplab \
#               --dataset segTHOR --metrics '[]'
# & mv ../genΩ