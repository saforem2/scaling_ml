#!/bin/bash
#COBALT -n 2
#COBALT -t 1:00:00 -q full-node
#COBALT -A SDL_Workshop -O results/thetagpu/$jobid.keras_mnist

#submisstion script for running tensorflow_mnist with horovod

echo "Running Cobalt Job $COBALT_JOBID."

#Loading modules

source /lus/theta-fs0/software/thetagpu/conda/tf_master/2020-11-11/mconda3/setup.sh

for n in 1 2 4 8
do
    mpirun -np $n python tensorflow2_keras_mnist.py --device gpu --epochs 32 >& results/thetagpu/tensorflow2_keras_mnist.out.$n & 
done
wait 

mpirun -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -np 16 -npernode 8 --hostfile ${COBALT_NODEFILE} python tensorflow2_keras_mnist.py --device gpu --epochs 32 >& results/thetagpu/tensorflow2_keras_mnist.out.16

