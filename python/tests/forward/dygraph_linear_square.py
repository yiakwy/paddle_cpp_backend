import numpy as np
import paddle

# 仍然用线性回归的例子
class SimpleFcLayer(paddle.nn.Layer):
    def __init__(self, batch_size, feature_size, fc_size):
        super(self.__class__, self).__init__()

        self._linear = paddle.nn.Linear(feature_size, fc_size)
        self._offset = paddle.to_tensor(
            np.random.random((batch_size, fc_size)).astype('float32'))

    def forward(self, x):
        fc = self._linear(x)
        return fc + self._offset

batch_size=3
feature_size=5
output_dim=1

shape = (batch_size, feature_size, output_dim)

fc_layer = SimpleFcLayer(*shape)

# 查验模型结构
paddle.summary(fc_layer, shape[0:2]) # 输出简单的线性结构
# ---------------------------------------------------------------------------
# Layer (type)       Input Shape          Output Shape         Param #
# ===========================================================================
#   Linear-5           [[3, 5]]              [3, 1]               6
# ===========================================================================
# Total params: 6
# Trainable params: 6
# Non-trainable params: 0
# ---------------------------------------------------------------------------
# Input size (MB): 0.00
# Forward/backward pass size (MB): 0.00
# Params size (MB): 0.00
# Estimated Total Size (MB): 0.00
# ---------------------------------------------------------------------------


in_data = np.random.random([batch_size, feature_size]).astype('float32')

# 将numpy的ndarray类型的数据转换为Tensor类型
x = paddle.to_tensor(in_data)

# 动态图执行方法 1 ：
# 训练模式下，即刻（imperative）执行
#   1. 会调用tracer创建反向传播的算子图
#   2. 立刻返回，异步获取计算结果
y = fc_layer.forward(x)

# call numpy() 打印结果
print("y:",y)
# y: Tensor(shape=[3, 1], dtype=float32, place=CPUPlace, stop_gradient=False,
#       [[1.69055665],
#        [1.21727896],
#        [1.11208010]])

# 动态图执行方法2:
# 预测模式，即 layer.trainable = False
fc_layer.eval()
y1 = fc_layer.forward(x)