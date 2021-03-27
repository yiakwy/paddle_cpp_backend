import numpy as np
import paddle

x_np = np.random.random([4, 5]).astype('float32')
x = paddle.to_tensor(x_np)

# 运行前向relu算子，记录反向relu信息
y = paddle.nn.functional.relu(x)
# 运行前向sum算子，记录反向sum信息
z = paddle.sum(y)
# 根据反向计算图执行反向
z.backward()