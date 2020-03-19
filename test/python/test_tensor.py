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
# =============================================================================
from __future__ import division

import math
import unittest
import numpy as np

from singa import tensor
from singa.proto import core_pb2

from cuda_helper import gpu_dev, cpu_dev


class TestTensorMethods(unittest.TestCase):

    def setUp(self):
        self.shape = (2, 3)
        self.t = tensor.Tensor(self.shape)
        self.s = tensor.Tensor(self.shape)
        self.t.set_value(0)
        self.s.set_value(0)

    def test_tensor_fields(self):
        t = self.t
        shape = self.shape
        self.assertTupleEqual(t.shape, shape)
        self.assertEqual(t.shape[0], shape[0])
        self.assertEqual(t.shape[1], shape[1])
        self.assertEqual(tensor.product(shape), 2 * 3)
        self.assertEqual(t.ndim(), 2)
        self.assertEqual(t.size(), 2 * 3)
        self.assertEqual(t.memsize(), 2 * 3 * tensor.sizeof(core_pb2.kFloat32))
        self.assertFalse(t.is_transpose())

    def test_unary_operators(self):
        t = self.t
        self.assertAlmostEqual(tensor.to_numpy(t)[0, 0], 0.0)
        t += 1.23
        self.assertAlmostEqual(tensor.to_numpy(t)[0, 0], 1.23)
        t -= 0.23
        self.assertAlmostEqual(tensor.to_numpy(t)[0, 0], 1.23 - 0.23)
        t *= 2.5
        self.assertAlmostEqual(tensor.to_numpy(t)[0, 0], (1.23 - 0.23) * 2.5)
        t /= 2
        self.assertAlmostEqual(
            tensor.to_numpy(t)[0, 0], (1.23 - 0.23) * 2.5 / 2)

    def test_binary_operators(self):
        t = self.t
        t += 3.2
        s = self.s
        s += 2.1
        a = t + s
        self.assertAlmostEqual(tensor.to_numpy(a)[0, 0], 3.2 + 2.1, 5)
        a = t - s
        self.assertAlmostEqual(tensor.to_numpy(a)[0, 0], 3.2 - 2.1, 5)
        a = t * s
        self.assertAlmostEqual(tensor.to_numpy(a)[0, 0], 3.2 * 2.1, 5)
        ''' not implemented yet
        a = t / s
        self.assertAlmostEqual(tensor.to_numpy(a)[0,0], 3.2/2.1, 5)
        '''

    def test_comparison_operators(self):
        t = self.t
        t += 3.45
        a = t < 3.45
        self.assertEqual(tensor.to_numpy(a)[0, 0], 0)
        a = t <= 3.45
        self.assertEqual(tensor.to_numpy(a)[0, 0], 1)
        a = t > 3.45
        self.assertEqual(tensor.to_numpy(a)[0, 0], 0)
        a = t >= 3.45
        self.assertEqual(tensor.to_numpy(a)[0, 0], 1)
        a = tensor.lt(t, 3.45)
        self.assertEqual(tensor.to_numpy(a)[0, 0], 0)
        a = tensor.le(t, 3.45)
        self.assertEqual(tensor.to_numpy(a)[0, 0], 1)
        a = tensor.gt(t, 3.45)
        self.assertEqual(tensor.to_numpy(a)[0, 0], 0)
        a = tensor.ge(t, 3.45)
        self.assertEqual(tensor.to_numpy(a)[0, 0], 1)

    def test_tensor_copy(self):
        t = tensor.Tensor((2, 3))
        t += 1.23
        self.assertAlmostEqual(tensor.to_numpy(t)[0, 0], 1.23)
        tc = t.copy()
        tdc = t.deepcopy()
        self.assertAlmostEqual(tensor.to_numpy(tc)[0, 0], 1.23)
        self.assertAlmostEqual(tensor.to_numpy(tdc)[0, 0], 1.23)
        t += 1.23
        self.assertAlmostEqual(tensor.to_numpy(t)[0, 0], 2.46)
        self.assertAlmostEqual(tensor.to_numpy(tc)[0, 0], 2.46)
        self.assertAlmostEqual(tensor.to_numpy(tdc)[0, 0], 1.23)

    def test_copy_data(self):
        t = self.t
        t += 1.23
        s = self.s
        s += 5.43
        self.assertAlmostEqual(tensor.to_numpy(t)[0, 0], 1.23)
        tensor.copy_data_to_from(t, s, 2)
        self.assertAlmostEqual(tensor.to_numpy(t)[0, 0], 5.43, 5)
        self.assertAlmostEqual(tensor.to_numpy(t)[0, 1], 5.43, 5)
        self.assertAlmostEqual(tensor.to_numpy(t)[0, 2], 1.23)

    def test_global_method(self):
        t = self.t
        t += 12.34
        a = tensor.log(t)
        self.assertAlmostEqual(tensor.to_numpy(a)[0, 0], math.log(12.34))

    def test_random(self):
        x = tensor.Tensor((1000,))
        x.gaussian(1, 0.01)
        self.assertAlmostEqual(tensor.average(x), 1, 3)

    def test_radd(self):
        x = tensor.Tensor((3,))
        x.set_value(1)
        y = 1 + x
        self.assertEqual(tensor.average(y), 2.)

    def test_rsub(self):
        x = tensor.Tensor((3,))
        x.set_value(1)
        y = 1 - x
        self.assertEqual(tensor.average(y), 0.)

    def test_rmul(self):
        x = tensor.Tensor((3,))
        x.set_value(1)
        y = 2 * x
        self.assertEqual(tensor.average(y), 2.)

    def test_rdiv(self):
        x = tensor.Tensor((3,))
        x.set_value(1)
        y = 2 / x
        self.assertEqual(tensor.average(y), 2.)

    def test_tensor_inplace_api(self):
        """ tensor inplace methods alter internal state and also return self
        """
        x = tensor.Tensor((3,))
        y = x.set_value(1)
        self.assertTrue(y is x)

        x = tensor.Tensor((3,))
        y = x.uniform(1, 2)
        self.assertTrue(y is x)

        x = tensor.Tensor((3,))
        y = x.bernoulli(1)
        self.assertTrue(y is x)

        x = tensor.Tensor((3,))
        y = x.gaussian(1, 2)
        self.assertTrue(y is x)

    def test_numpy_convert(self):
        a = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.int)
        t = tensor.from_numpy(a)
        b = tensor.to_numpy(t)
        self.assertEqual(np.sum(a - b), 0)

        a = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        t = tensor.from_numpy(a)
        b = tensor.to_numpy(t)
        self.assertEqual(np.sum(a - b), 0.)

    def test_transpose(self):
        a = np.array(
            [1.1, 1.1, 1.1, 1.1, 1.4, 1.3, 1.1, 1.6, 1.1, 1.1, 1.1, 1.2])
        a = np.reshape(a, (2, 3, 2))
        ta = tensor.from_numpy(a)

        A1 = np.transpose(a)
        tA1 = tensor.transpose(ta)
        TA1 = tensor.to_numpy(tA1)
        A2 = np.transpose(a, [0, 2, 1])
        tA2 = tensor.transpose(ta, [0, 2, 1])
        TA2 = tensor.to_numpy(tA2)

        np.testing.assert_array_almost_equal(TA1, A1)
        np.testing.assert_array_almost_equal(TA2, A2)

    def test_einsum(self):

        a = np.array(
            [1.1, 1.1, 1.1, 1.1, 1.4, 1.3, 1.1, 1.6, 1.1, 1.1, 1.1, 1.2])
        a = np.reshape(a, (2, 3, 2))
        ta = tensor.from_numpy(a)

        res1 = np.einsum('kij,kij->kij', a, a)
        tres1 = tensor.einsum('kij,kij->kij', ta, ta)
        Tres1 = tensor.to_numpy(tres1)
        res2 = np.einsum('kij,kih->kjh', a, a)
        tres2 = tensor.einsum('kij,kih->kjh', ta, ta)
        Tres2 = tensor.to_numpy(tres2)

        self.assertAlmostEqual(np.sum(Tres1 - res1), 0., places=3)
        self.assertAlmostEqual(np.sum(Tres2 - res2), 0., places=3)

    def test_repeat(self):

        a = np.array(
            [1.1, 1.1, 1.1, 1.1, 1.4, 1.3, 1.1, 1.6, 1.1, 1.1, 1.1, 1.2])
        a = np.reshape(a, (2, 3, 2))
        ta = tensor.from_numpy(a)

        ta_repeat1 = tensor.repeat(ta, 2, axis=None)
        a_repeat1 = np.repeat(a, 2, axis=None)
        Ta_repeat1 = tensor.to_numpy(ta_repeat1)
        ta_repeat2 = tensor.repeat(ta, 4, axis=1)
        a_repeat2 = np.repeat(a, 4, axis=1)
        Ta_repeat2 = tensor.to_numpy(ta_repeat2)

        self.assertAlmostEqual(np.sum(Ta_repeat1 - a_repeat1), 0., places=3)
        self.assertAlmostEqual(np.sum(Ta_repeat2 - a_repeat2), 0., places=3)

    def test_sum(self):
        a = np.array(
            [1.1, 1.1, 1.1, 1.1, 1.4, 1.3, 1.1, 1.6, 1.1, 1.1, 1.1, 1.2])
        a = np.reshape(a, (2, 3, 2))
        ta = tensor.from_numpy(a)

        a_sum0 = np.sum(a)
        ta_sum0 = tensor.sum(ta)
        Ta_sum0 = tensor.to_numpy(ta_sum0)
        a_sum1 = np.sum(a, axis=1)
        ta_sum1 = tensor.sum(ta, axis=1)
        Ta_sum1 = tensor.to_numpy(ta_sum1)
        a_sum2 = np.sum(a, axis=2)
        ta_sum2 = tensor.sum(ta, axis=2)
        Ta_sum2 = tensor.to_numpy(ta_sum2)

        self.assertAlmostEqual(np.sum(a_sum0 - Ta_sum0), 0., places=3)
        self.assertAlmostEqual(np.sum(a_sum1 - Ta_sum1), 0., places=3)
        self.assertAlmostEqual(np.sum(a_sum2 - Ta_sum2), 0., places=3)

    def test_tensordot(self):
        a = np.array(
            [1.1, 1.1, 1.1, 1.1, 1.4, 1.3, 1.1, 1.6, 1.1, 1.1, 1.1, 1.2])
        a = np.reshape(a, (2, 3, 2))

        ta = tensor.from_numpy(a)

        res1 = np.tensordot(a, a, axes=1)
        tres1 = tensor.tensordot(ta, ta, axes=1)
        Tres1 = tensor.to_numpy(tres1)
        self.assertAlmostEqual(np.sum(Tres1 - res1), 0., places=3)
        np.testing.assert_array_almost_equal(Tres1, res1)

        res2 = np.tensordot(a, a, axes=([0, 1], [2, 1]))
        tres2 = tensor.tensordot(ta, ta, axes=([0, 1], [2, 1]))
        np.testing.assert_array_almost_equal(tensor.to_numpy(tres2), res2)

    def test_reshape(self):
        a = np.array([[[1.1, 1.1, 1.4], [1.1, 1.1, 1.1]],
                      [[1.1, 1.1, 1.3], [1.6, 1.1, 1.2]]])
        ta = tensor.from_numpy(a)
        tb = tensor.reshape(ta, [2, 6])
        self.assertAlmostEqual(tb.shape[0], 2., places=3)
        self.assertAlmostEqual(tb.shape[1], 6., places=3)
        np.testing.assert_array_almost_equal(tensor.to_numpy(tb),
                                             a.reshape((2, 6)))

    def test_transpose_then_reshape(self):
        a = np.array([[[1.1, 1.1], [1.1, 1.1], [1.4, 1.3]],
                      [[1.1, 1.6], [1.1, 1.1], [1.1, 1.2]]])
        TRANSPOSE_AXES = (2, 0, 1)
        RESHAPE_DIMS = (2, 6)

        ta = tensor.from_numpy(a)
        ta = ta.transpose(TRANSPOSE_AXES)
        ta = ta.reshape(RESHAPE_DIMS)

        np.testing.assert_array_almost_equal(
            tensor.to_numpy(ta),
            np.reshape(a.transpose(TRANSPOSE_AXES), RESHAPE_DIMS))

    def test_concatenate(self):
        np1 = np.random.random([5, 6, 7, 8]).astype(np.float32)
        np2 = np.random.random([5, 6, 7, 1]).astype(np.float32)
        np3 = np.concatenate((np1, np2), axis=3)

        for dev in [cpu_dev, gpu_dev]:
            t1 = tensor.Tensor(device=dev, data=np1)
            t2 = tensor.Tensor(device=dev, data=np2)

            t3 = tensor.concatenate((t1, t2), 3)

            np.testing.assert_array_almost_equal(tensor.to_numpy(t3), np3)

    def test_subscription_cpu(self):
        np1 = np.random.random((5, 5, 5, 5)).astype(np.float32)
        sg_tensor = tensor.Tensor(device=cpu_dev, data=np1)
        sg_tensor_ret = sg_tensor[1:3, :, 1:, :-1]
        np.testing.assert_array_almost_equal((tensor.to_numpy(sg_tensor_ret)),
                                             np1[1:3, :, 1:, :-1])

    def test_subscription_gpu(self):
        np1 = np.random.random((5, 5, 5, 5)).astype(np.float32)
        sg_tensor = tensor.Tensor(device=gpu_dev, data=np1)
        sg_tensor_ret = sg_tensor[1:3, :, 1:, :-1]
        np.testing.assert_array_almost_equal((tensor.to_numpy(sg_tensor_ret)),
                                             np1[1:3, :, 1:, :-1])

    def test_ceil(self):

        for dev in [cpu_dev, gpu_dev]:

            np1 = np.random.random([5, 6, 7, 8]).astype(np.float32)
            np1 = np1 * 10
            np2 = np.ceil(np1)

            t1 = tensor.Tensor(device=dev, data=np1)

            t2 = tensor.ceil(t1)

            np.testing.assert_array_almost_equal(tensor.to_numpy(t2), np2)

    def test_astype(self):
        for dev in [cpu_dev, gpu_dev]:
            shape1 = [2, 3]
            shape2 = [3, 2]

            np_flt = np.random.random(shape1).astype(np.float32)
            np_flt = np_flt * 10 - 5

            np_int = np_flt.astype(np.int32)
            np_flt2 = np_int.astype(np.float32)

            t2 = tensor.Tensor(device=dev, data=np_flt)
            t2 = t2.as_type('int')
            np.testing.assert_array_almost_equal(tensor.to_numpy(t2), np_int)

            t1 = t2.reshape(shape2)
            np.testing.assert_array_almost_equal(tensor.to_numpy(t1),
                                                 np_int.reshape(shape2))

            t1 = t1.as_type('float')
            np.testing.assert_array_almost_equal(tensor.to_numpy(t1),
                                                 np_flt2.reshape(shape2))


if __name__ == '__main__':
    unittest.main()
