#!/bin/bash

# Set job name
#SBATCH --job-name=OCTCubeM-OCTCube
# Specify the number of nodes and processors and gpus per nodes
#SBATCH --nodes=1 --ntasks-per-node=1 --gpus-per-node=1
#SBATCH --cpus-per-task=17


# For ascend cluster, we have nextgen and quad nodes
#SBATCH --partition=nextgen


# Specify the amount of time for this job
#SBATCH --time=36:00:00

# Specify the maximum amount of physical memory required
#SBATCH --mem=128gb

# Specify an account when more than one available
#SBATCH --account=PCON0023

#SBATCH --output=log_pt/%j_0_log.out

#SBATCH --error=log_pt/%j_0_log.err


# Load modules:
module load cuda/11.8.0

module load miniconda3/24.1.2-py310

source activate octcube

cd /fs/ess/PCON0023/shileicao/code/OCTCubeM/OCTCube

HOME=/fs/ess/PCON0023/shileicao/code
k_folds=5
OUTPUT_DIR=./outputs_ft_st/finetune_inhouse_multi_label_3D_correct_patient_singlefold/

python -m main_finetune_downstream_inhouse_singlefold --nb_classes 8 \
    --data_path /fs/ess/PCON0023/eye3d/data/ukbiobank/oct \
    --task ${OUTPUT_DIR} \
    --single_fold \
    --k_folds $k_folds \
    --num_frames 48 \
    --split_path $HOME/OCTCubeM/assets/Oph_cls_task/scr_train_val_test_split_622/ \
    --task_mode multi_label  \
    --enable_early_stop \
    --early_stop_patience 8 \
    --val_metric AUC \
    --input_size 256 \
    --log_dir log_pt/ \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 1 \
    --warmup_epochs 2 \
    --world_size 1 \
    --model flash_attn_vit_large_patch16 \
    --patient_dataset_type 3D_st_flash_attn \
    --transform_type monai_3D \
    --color_mode gray \
    --epochs 20 \
    --blr 5e-3 \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.2 \
    --always_test \
    --finetune $HOME/OCTCubeM/ckpt/OCTCube.pth \
    --return_bal_acc \
    # --resume latest \
    # --save_model \
