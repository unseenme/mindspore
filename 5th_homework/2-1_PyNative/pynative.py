# encoding: utf-8

import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor
 
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
 
fc = nn.Dense(3, 4)
input_data = Tensor(np.ones([2, 3]).astype(np.float32))
output = fc(input_data)
print(output.asnumpy())
