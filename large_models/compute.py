import os

import random


def get_gpu_memory():
    lines = os.popen('nvidia-smi -q -d Memory |grep -A5 GPU|grep Free').readlines()

    memory_available = [int(x.split()[2]) for x in lines]
    gpus = {index: mb for index, mb in enumerate(memory_available)}
    return gpus


def get_random_with_gpu_with_gb_free(gb):
    gpus = get_gpu_memory()
    filtered_keys = [key for key, value in gpus.items() if value // 1000 > gb]
    if filtered_keys:
        return random.choice(filtered_keys)
    else:
        return None


def get_device():
    import torch

    def get_index_of_free_gpus():

        gpus = get_gpu_memory()

        gpus = {index: mega for index, mega in gpus.items()}
        gpus = {k: v for k, v in sorted(gpus.items(), key=lambda x: x[1], reverse=True)}

        return gpus

    def compute_gpu_indent(gpus):
        try:
            # return 'cuda'
            best_gpu = max(gpus, key=lambda gpu_num: gpus[int(gpu_num)])
            return 'cuda:' + str(best_gpu)
        except:
            return 'cuda'

    gpus = get_index_of_free_gpus()
    # print(gpus)
    device = compute_gpu_indent(gpus) if torch.cuda.is_available() else 'cpu'
    print('device:', device)
    return torch.device(device)
