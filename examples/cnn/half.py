from singa import opt
from singa import device
from singa import tensor
import subprocess as sp
import os

def get_gpu_memory():
  _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

  ACCEPTABLE_AVAILABLE_MEMORY = 1024
  COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
  memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
  memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
  print(memory_free_values)
  return memory_free_values


dev_id = 7
dev = device.create_cuda_gpu_on(dev_id)
dtype=tensor.float32
batch_size=128
IMG_SIZE=224

gmem_old = get_gpu_memory()[dev_id]


all_tx = []
for i in range(10):
    tx = tensor.Tensor((100000000,), dev, dtype)
    tx.set_value(1.0)
    # all_tx.append(tx)
    dev.Sync()

    gmem_new = get_gpu_memory()[dev_id]
    print("gmem", gmem_new-gmem_old)
    gmem_old=gmem_new