##!/bin/bash

#######################################################################################################
# General Evaluation
#######################################################################################################
for target in INFERENCE_TIME POWER_CONSUMPTION CPU_MEM CPU_MEM_PEAK CPU_LOAD GPU_MEM GPU_MEM_PEAK GPU_LOAD
do
    python3 general-evaluation.py --target $target
done

#######################################################################################################
# Task Reduction
#######################################################################################################
for target in INFERENCE_TIME POWER_CONSUMPTION CPU_MEM CPU_MEM_PEAK CPU_LOAD GPU_MEM GPU_MEM_PEAK GPU_LOAD
do
    python3 task-reduction.py --target $target --task_red_index 100 
    python3 task-reduction.py --target $target --task_specific bodypose-estimation
    python3 task-reduction.py --target $target --task_specific face-landmarks-detection
done

#######################################################################################################
# Platform Reduction
#######################################################################################################
for target in INFERENCE_TIME POWER_CONSUMPTION CPU_MEM CPU_MEM_PEAK CPU_LOAD GPU_MEM GPU_MEM_PEAK GPU_LOAD
do
    python3 platform-reduction.py --target $target --platform_red_index 300 
    python3 platform-reduction.py --target $target --platform_specific jetson-xav-agx
    python3 platform-reduction.py --target $target --platform_specific jetson-nano2
done

for target in INFERENCE_TIME CPU_MEM CPU_MEM_PEAK CPU_LOAD 
do
    python3 platform-reduction.py --target $target --platform_specific cpu-only
done
