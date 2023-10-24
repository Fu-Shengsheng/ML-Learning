import numpy as np
import os
import subprocess
from dezero.core_simple import Variable, Function

# 微分法求导
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

# 传入一个 Variable 实例，返回 DOT 格式的字符串
def _dot_var(v: Variable, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.data.dtype)
    # 通过内置 id 方法生成唯一 id
    return dot_var.format(id(v), name)

# 传入一个 Function 实例，返回 DOT 格式字符串
def _dot_func(f: Function):
    # func 节点定义
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    # 入参 -> func -> 出参 连接的定义
    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        # y 是 weakref, 故此处应该使用 y() 获取 y 的真实值，不然计算 id 会与作为 input 时不同，导致计算图中断
        txt += dot_edge.format(id(f), id(y()))

    return txt

# 可视化计算图
def get_dot_graph(output: Variable, verbose=True):
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)
    
    add_func(output.creator)
    # 只需要计算最后输出的 output
    # 其余的 Variable 节点都由 Function 实例的 input 生成
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)

    return 'digraph g {\n' + txt + '}'

def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)

    # 将 DOT 数据保存至文件
    # 文件目录为 ~/.dezero
    # os.path.expanduser('~') 的含义是展开主目录路径 ~
    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    # 文件名
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    # 计算导出文件的扩展名，如 png, pdf 等
    extension = os.path.splitext(to_file)[1][1:]
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)