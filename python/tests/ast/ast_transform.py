import textwrap
import ast
# 为了应对不同版本的AST(py33,py36 ...)，我们采用gast
import gast
import astor
# 基于虚拟机的语言都保存有源代码，inspect是获取源代码的快捷操作
import inspect
import paddle
from paddle.fluid.dygraph.dygraph_to_static.origin_info import attach_origin_info
from paddle.fluid.dygraph.dygraph_to_static.utils import ast_to_func
from paddle.fluid.dygraph.dygraph_to_static.ast_transformer import DygraphToStaticAst
import numpy as np
from paddle.jit import TracedLayer

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

## Translator AST analysis
def pyfunc_to_ast(pyfunc):
    """
    Transform py func to AST node
    """
    src_code = textwrap.dedent(inspect.getsource(pyfunc))
    ast_root = gast.parse(src_code)
    return ast_root

def ast_to_code(ast_node):
    """
    Transforms ast node into source code.
    """
    if not isinstance(ast_node, (gast.AST, ast.AST)):
        raise TypeError(
            "Type of ast_root should be gast.AST or ast.AST, but received %s." %
            type(ast_node))
    if isinstance(ast_node, gast.AST):
        ast_node = gast.gast_to_ast(ast_node)
    code = astor.to_source(ast_node)
    return code

batch_size=3
feature_size=5
output_dim=1

shape = (batch_size, feature_size, output_dim)

fc_layer = SimpleFcLayer(*shape)

#  Using Translator API
# prog_translator = paddle.jit.ProgramTranslator()
# y = prog_translator.get_output(fc_layer.forward, x)
# print("y:", y)
# print()

# build ast once or fetch it from cache
ast_root = pyfunc_to_ast(fc_layer.forward)

# Transform AST
dygraph_to_static = DygraphToStaticAst()

# root = ast_root
# func = fc_layer.forward

# root = attach_origin_info(root, func)

# root_wrapper = dygraph_to_static.get_static_ast(root)
root_wrapper = dygraph_to_static.get_static_ast(ast_root)
transformed_code = ast_to_code(root_wrapper.node)

print(transformed_code)
# 转写后的代码：
# def forward(self, x):
#    fc = paddle.jit.dy2static.convert_call(self._linear)(x)
#    return fc + self._offset

converted_call = paddle.jit.dy2static.convert_call(fc_layer._linear)
print(converted_call)
# Linear(in_features=5, out_features=1, dtype=float32)
# We just get static graph structure here !
# Only what we need to do now is to add the structure to paddle.fluid program and the default block!