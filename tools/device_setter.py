from mimetypes import init
import os
import torch
from concurrent.futures import ProcessPoolExecutor

def initializer(q):
    x = q.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = x

def multi_GPU_executor(tasks_per_gpu = 1, gpus = [], initfn = initializer):
    q = torch.multiprocessing.Queue()

    num_gpus = len(gpus)
    assert num_gpus > 0
    executor = ProcessPoolExecutor(max_workers = tasks_per_gpu * num_gpus, mp_context = torch.multiprocessing, initializer = initfn, initargs = (q,))
    gpus = [str(i) for i in gpus]
    gpus = tasks_per_gpu * gpus
    [q.put(i) for i in gpus]
    return executor