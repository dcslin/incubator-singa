#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

from singa import tensor
from singa.tensor import Tensor
from singa import autograd
from singa import opt
from singa import device
import numpy as np

if __name__ == "__main__":

    np.random.seed(0)

    autograd.training = True

    # prepare training data in numpy array

    # generate the boundary
    f = lambda x: (5 * x + 1)
    bd_x = np.linspace(-1.0, 1, 200)
    bd_y = f(bd_x)
    # generate the training data
    x = np.random.uniform(-1, 1, 400)
    y = f(x) + 2 * np.random.randn(len(x))
    # convert training data to 2d space
    label = np.asarray([5 * a + 1 > b for (a, b) in zip(x, y)])
    data = np.array([[a, b] for (a, b) in zip(x, y)], dtype=np.float32)

    def to_categorical(y, num_classes):
        """
        Converts a class vector (integers) to binary class matrix.

        Args
            y: class vector to be converted into a matrix
                (integers from 0 to num_classes).
            num_classes: total number of classes.

        Return
            A binary matrix representation of the input.
        """
        y = np.array(y, dtype="int")
        n = y.shape[0]
        categorical = np.zeros((n, num_classes))
        categorical[np.arange(n), y] = 1
        return categorical

    label = to_categorical(label, 2).astype(np.float32)
    print("train_data_shape:", data.shape)
    print("train_label_shape:", label.shape)
    # print(data[:2])
    # print(label[:2])

    precision=tensor.float16
    np_precision = np.float16

    # precision=tensor.float32
    # np_precision = np.float32

    dev = device.create_cuda_gpu()

    inputs = Tensor(data=data,device=dev)
    target = Tensor(data=label,device=dev)

    inputs = inputs.as_type(precision)
    target = target.as_type(tensor.int32)

    assert target.dtype == tensor.int32
    print(inputs.dtype)
    print(target.dtype)

    w0_np = np.random.normal(0, 0.1, (2,3)).astype(np_precision)
    w0 = Tensor(data=w0_np,device=dev, dtype=precision, requires_grad=True, stores_grad=True)
    print("w0:",np.linalg.norm(tensor.to_numpy(w0)))
    # w0.gaussian(0.0, 0.1)
    b0 = Tensor(shape=(3,),device=dev, dtype=precision, requires_grad=True, stores_grad=True)
    b0.set_value(0.0)

    w1_np = np.random.normal(0, 0.1, (3,2)).astype(np_precision)
    w1 = Tensor(data=w1_np,device=dev, dtype=precision, requires_grad=True, stores_grad=True)
    # w1 = Tensor(shape=(3, 2),device=dev, dtype=precision, requires_grad=True, stores_grad=True)
    # w1.gaussian(0.0, 0.1)
    b1 = Tensor(shape=(2,),device=dev, dtype=precision, requires_grad=True, stores_grad=True)
    b1.set_value(0.0)
    # print(w0,b0,w1,b1)

    sgd = opt.SGD(0.1)
    # training process
    for i in range(1001):
        # print("0:",np.linalg.norm(tensor.to_numpy(inputs)))
        x = autograd.matmul(inputs, w0)
        # print("1:",np.linalg.norm(tensor.to_numpy(x)))
        x = autograd.add_bias(x, b0)
        # print("2:",np.linalg.norm(tensor.to_numpy(x)))
        x = autograd.relu(x)
        # print("3:",np.linalg.norm(tensor.to_numpy(x)))
        x = autograd.matmul(x, w1)
        # print("4:",np.linalg.norm(tensor.to_numpy(x)))
        x = autograd.add_bias(x, b1)
        # print("5:",np.linalg.norm(tensor.to_numpy(x)))
        loss = autograd.softmax_cross_entropy(x, target)
        # x2 = x.as_type(tensor.float32)
        # loss2 = autograd.softmax_cross_entropy(x2, target)
        # print(loss)
        # print(loss2)
        sgd(loss)
        # grads = autograd.gradients(loss)
        # for k,v in grads.items():
        #     assert v.dtype == precision

        if i % 100 == 0:
            print("training loss = ", tensor.to_numpy(loss)[0])

        # print("grads",grads)

        # print("6 w0:",np.linalg.norm(tensor.to_numpy(w0)))
