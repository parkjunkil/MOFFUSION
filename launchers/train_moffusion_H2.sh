# Reference: The code has been modified from the LDM repo: https://https://github.com/yccyenchicheng/SDFusion

RED="\033[0;31m"
NC="\033[0m" # No Color
DATE_WITH_TIME=`date "+%Y-%m-%dT%H-%M-%S"`

logs_dir='./outputs/logs'
if [ `hostname` = az007 ] || [ `hostname` = az006 ]; then
    logs_dir="logs_home"
fi

### set gpus ###
gpu_ids=0         # single-gpu
# gpu_ids=0,1,2,3  # multi-gpu

if [ ${#gpu_ids} -gt 1 ]; then
    # specify these two if multi-gpu
    # NGPU=2
    # NGPU=3
    NGPU=4
    PORT=11768
    echo "HERE"
fi
################

### hyper params ###
lr=1e-4
batch_size=16
####################

### model stuff ###
model="moffusion_H2"
df_cfg="./configs/moffusion-H2.yaml"

vq_model="vqvae_moffusion"
vq_cfg="./configs/vqvae.yaml"
vq_ckpt="./saved_ckpt/vqvae.pth"
vq_dset="mof_250k"
###################

### dataset stuff ###
max_dataset_size=10000000
dataset_mode='mof_250k'
dataroot="./data"
res=32
trunc_thres=0.2
#####################

### display & log stuff ##
display_freq=50
print_freq=50
total_iters=100000000
save_steps_freq=50
##########################


today=$(date "+%m%d")
me=`basename "$0"`
me=$(echo $me | cut -d"." -f 1)

note="moffusion-H2"

name="${DATE_WITH_TIME}-${model}-${dataset_mode}-LR${lr}-${note}"

debug=0
if [ $debug = 1 ]; then
    printf "${RED}Debugging!${NC}\n"
	max_dataset_size=1000
    total_iters=1000000
	display_freq=6
    save_steps_freq=${display_freq}
	print_freq=2
    name="DEBUG-${name}"
fi


cmd="train.py --name ${name} --logs_dir ${logs_dir} --gpu_ids ${gpu_ids} --lr ${lr} --batch_size ${batch_size} \
    --model ${model} --df_cfg ${df_cfg} \
    --vq_model ${vq_model} --vq_cfg ${vq_cfg} --vq_ckpt ${vq_ckpt} --vq_dset ${vq_dset}\
    --dataset_mode ${dataset_mode}\
    --max_dataset_size ${max_dataset_size} --trunc_thres ${trunc_thres} \
    --display_freq ${display_freq} --print_freq ${print_freq} \
    --total_iters ${total_iters} --save_steps_freq ${save_steps_freq} \
    --debug ${debug}"

if [ ! -z "$dataroot" ]; then
    cmd="${cmd} --dataroot ${dataroot}"
    echo "setting dataroot to: ${dataroot}"
fi

if [ ! -z "$ckpt" ]; then
    cmd="${cmd} --ckpt ${ckpt}"
    echo "continue training with ckpt=${ckpt}"
fi

if [ ! -z "$backend" ]; then
    cmd="${cmd} --backend ${backend}"
fi

multi_gpu=0
if [ ${#gpu_ids} -gt 1 ]; then
    multi_gpu=1
fi

echo "[*] Training is starting on `hostname`, GPU#: ${gpu_ids}, logs_dir: ${logs_dir}"

if [ $multi_gpu = 1 ]; then
    cmd="-m torch.distributed.launch --nproc_per_node=${NGPU} --master_port=${PORT} ${cmd}"
fi

echo "[*] Training with command: "
echo "CUDA_VISIBLE_DEVICES=${gpu_ids} python ${cmd}"

# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=${gpu_ids} python ${cmd}
CUDA_VISIBLE_DEVICES=${gpu_ids} python ${cmd}
