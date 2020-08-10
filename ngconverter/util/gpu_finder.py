import os
import numpy as np

def get_free_gpu_list():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('gpu_stat_tmp', 'r').readlines()]
    os.system('rm gpu_stat_tmp')
    return memory_available

def get_best_gpu(candidate_list):
    memory_available = get_free_gpu_list()
    if candidate_list is None:
        return str(np.argmax(memory_available))
    candidate_usages = [memory_available[int(c)] for c in candidate_list]
    return candidate_list[np.argmax(candidate_usages)]

