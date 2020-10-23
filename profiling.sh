#!/usr/bin/env bash
rm profile*txt
python torch-examples/cnn_mnist.py --opt-level=O3 > profile_torch_cnn_mnist_fp16.txt
python torch-examples/cnn_mnist.py --opt-level=O0 > profile_torch_cnn_mnist_fp32.txt
python examples/cnn/train_cnn.py cnn mnist -pfloat32 -m1 > profile_singa_cnn_mnist_fp32.txt
python examples/cnn/train_cnn.py cnn mnist -pfloat16 -m1 > profile_singa_cnn_mnist_fp16.txt