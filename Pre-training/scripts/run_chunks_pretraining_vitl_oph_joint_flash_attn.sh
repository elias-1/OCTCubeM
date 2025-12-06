#!/bin/bash

# Set job name
#SBATCH --job-name=med_oct_foundation_run1
# Specify the number of nodes and processors and gpus per nodes
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1
#SBATCH --cpus-per-task=17


# For ascend cluster, we have nextgen and quad nodes
#SBATCH --partition=nextgen


# Specify the amount of time for this job
#SBATCH --time=96:00:00

# Specify the maximum amount of physical memory required
#SBATCH --mem=128gb

# Specify an account when more than one available
#SBATCH --account=PCON0023


# Load modules:
module load cuda/11.8.0

module load miniconda3/24.1.2-py310

source activate octcube

cd /fs/ess/PCON0023/shileicao/code/OCTCubeM/Pre-training


OUTPUTDIR=pretrain-output
prefix=oct
DATA_PATH=/fs/ess/PCON0023/eye3d/data/ukbiobank/oct
kermany_data_dir=/fs/ess/PCON0023/eye3d/data/CellData/OCT/
pretrain_type=training_new
init_ckpt_path=OCTCubeM/ckpt/OCTCube.pth

BSZ=4
INPUTSIZE=256
ACCUMSTEPS=1
EPOCHS=50
BLR=1.6e-3
RATIO=0.9


SCHEDULER="bsz-$BSZ-inputsize-$INPUTSIZE-aacumsteps$ACCUMSTEPS-ep-$EPOCHS-lr-$BLR-2d512-flash-attn-$pretrain_type"
OUTPUTDIR=$OUTPUTDIR/$SCHEDULER

python run_pretrain_oph_joint_2d512_flash_attn.py \
        --data_path $DATA_PATH \
        --output_dir $OUTPUTDIR \
        --kermany_data_dir $kermany_data_dir \
        --log_dir $OUTPUTDIR/log_dir \
        --batch_size $BSZ \
        --accum_iter $ACCUMSTEPS \
        --epochs $EPOCHS \
        --blr $BLR \
        --mask_ratio $RATIO \
        --weight_decay 0.05 \
        --num_workers 24 \
        --num_frames 60 \
        --t_patch_size 3 \
        --pred_t_dim 60 \
        --input_size $INPUTSIZE \
        --warmup_epochs 1 \
        --init_ckpt $init_ckpt_path \
        --resume_type ${pretrain_type} \
        --model flash_attn_mae_vit_large_patch16 \
        --batch_size_2d 64 \
        --mask_ratio_2d_min 0.75 \
        --mask_ratio_2d_max 0.85 \
        --K_min 0.15 \
        --K_max 0.3 \
        --epoch_offset 0 \
        --high_res_input_size 512 \


