"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
@author: Andreas Doerr
"""

import numpy as np
import tensorflow as tf


def backward(y):
    assert not np.any(y <= 1e-10), 'Input to backward transformation should be greater 1e-10'
    result = np.log(np.exp(y - 1e-10) - np.ones(1))
    return np.where(y > 35, y-1e-10, result)


def forward(x):
    x = tf.convert_to_tensor(x)
    return tf.nn.softplus(x) + 1e-10
