srun -K  --job-name=activ --partition=RTXA6000,RTXA6000-EI --gpus=1 --cpus-per-task=2 --mem=40G --time=2-00:00 --output output/console.%A_%a.out --error output/console.%A_%a.error  \
          --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_23.08-py3.sqsh \
          --container-workdir=`pwd` \
          --container-mounts=/netscratch:/netscratch,/ds:/ds,`pwd`:`pwd` \
          --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
          install.sh python3 train.py "$@" &
