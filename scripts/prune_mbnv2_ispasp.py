# run i-SpaSP pruning on resnet34 for imagenet

import os

GPU = 0
SAVE_DIR = './results/'
EXP_NAME = 'ispasp_mbnv2_imn_00'
DATA_PATH = '' # path to imagenet dataset
BATCH_SIZE = 256
CS_BATCHES = 5
CS_ITER = 20
BLOCK_FT_ES = 1
BLOCK_FT_LR = 0.01
PRUNE_FT_ES = 90
PRUNE_FT_LR = 0.01
USE_LR_SCHED = True
PRUNED_PATH = None # path to pruned model checkpoint
VERBOSE = False

command = (
    f'CUDA_VISIBLE_DEVICES={GPU} python prune_mobilenetv2_ispasp.py --save-dir {SAVE_DIR} '
    f'--exp-name {EXP_NAME} --data-path {DATA_PATH} --batch-size {BATCH_SIZE} '
    f'--num-cs-batches {CS_BATCHES} --num-cs-iter {CS_ITER} --block-ft-epochs {BLOCK_FT_ES} '
    f'--block-ft-lr {BLOCK_FT_LR} --prune-ft-epochs {PRUNE_FT_ES} --prune-ft-lr {PRUNE_FT_LR}')
if PRUNED_PATH is not None:
    command += f' --pruned-path {PRUNED_PATH}'
if USE_LR_SCHED:
    command += f' --use-lr-sched'
if VERBOSE:
    command += f' --verbose'
os.system(command) 
