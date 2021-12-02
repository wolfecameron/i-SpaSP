# run i-SpaSP pruning on resnet34 for imagenet

import os

GPU = 1
SAVE_DIR = './results/'
EXP_NAME = 'ispasp_rn34_imn_00'
DATA_PATH = '' # put path to imagenet here
BATCH_SIZE = 256
CS_BATCHES = 5
CS_ITER = 20
RATIOS = [0.4, 0.4, 0.4, 1.0]
BLOCK_FT_ES = 1
BLOCK_FT_LR = 0.01
PRUNE_FT_ES = 90
PRUNE_FT_LR = 0.01
PRUNED_PATH = None # path to pruned model checkpoint
USE_LR_SCHED = True
PRUNE_LAST_LAYER = False
VERBOSE = False 

command = (
    f'CUDA_VISIBLE_DEVICES={GPU} python prune_resnet34_ispasp.py --save-dir {SAVE_DIR} '
    f'--exp-name {EXP_NAME} --data-path {DATA_PATH} --batch-size {BATCH_SIZE} '
    f'--num-cs-batches {CS_BATCHES} --num-cs-iter {CS_ITER} --layer1-ratio {RATIOS[0]} '
    f'--layer2-ratio {RATIOS[1]} --layer3-ratio {RATIOS[2]} --layer4-ratio {RATIOS[3]} '
    f'--block-ft-epochs {BLOCK_FT_ES} --block-ft-lr {BLOCK_FT_LR} --prune-ft-epochs {PRUNE_FT_ES} '
    f'--prune-ft-lr {PRUNE_FT_LR} ')
if PRUNE_LAST_LAYER:
    command += f' --prune-last-layer'
if PRUNED_PATH is not None:
    command += f' --pruned-path {PRUNED_PATH}'
if USE_LR_SCHED:
    command += ' --use-lr-sched'
if VERBOSE:
    command += ' --verbose'
os.system(command) 
