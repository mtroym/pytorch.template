#!/bin/bash
export ANACONDA=/mnt/cephfs_wj/mlnlp/tony/venv/.anaconda2/bin/
export PATH=$PATH:$ANACONDA
echo $PATH
export SRCPATH=/mnt/cephfs_wj/mlnlp/tony/src/
cd $SRCPATH
source activate py27