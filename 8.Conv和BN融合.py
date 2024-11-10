import time
import torch
import torch.nn as nn
import torchvision

# dummy:占位符, 在接收输入 x 后直接返回
class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()
    def forward(self, x):
        return x
    
def fuse(conv, bn):
    w = conv.weight
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    if conv.bias is not None:
        b = conv.bias
    else:
        b = torch.zeros_like(mean)
    
    # 权重融合 
    w = w * (gamma / var_sqrt).reshape([conv.out_channels, 1, 1, 1])    # 方便广播
    b = ((b - mean) * gamma) / var_sqrt + beta
    fuse_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=True)
    fuse_conv.weight = nn.Parameter(w)
    fuse_conv.bias = nn.Parameter(b)

    return fuse_conv

# 递归遍历神经网络模块m 中所有子模块，将卷积层和批归一化层融合成一个新的卷积层，用占位符DummyModule替换批归一化层
def fuse_module(m):
    children = list(m.named_children()) # 获取m的所有子模块, 并存储为(name, child)的列表
    conv = None                         # 存储当前卷积层
    conv_name = None                    # 存储当前卷积层名字
    
    for name, child in children:
        # isinstance(object, classinfo): 检查一个对象是否是指定类型或其子类型的实例
        if isinstance(child, nn.BatchNorm2d) and conv:
            bc = fuse(conv, child)        
            m._modules[conv_name] = bc        # 将融合后的模块替换掉原来的Conv2d层
            m._modules[name] = DummyModule() # 用DummyModule替换原来的BatchNorm2d
            conv = None
        elif isinstance(child, nn.Conv2d):
            conv = child
            conv_name = name
        else:
            # 如果既不是Conv2d也不是BatchNorm2d，则递归处理子模块
            fuse_module(child)

def test_net(m):
    p = torch.randn([1, 3, 224, 224])
    s = time.time()
    o_output = m(p)
    print("original time:", time.time() - s)
    
    fuse_module(m)
    s = time.time()
    f_output = m(p)
    print("fuse time", time.time() - s)
    print("max abs diff", (o_output - f_output).abs().max().item())

def test_layer(m):
    p = torch.randn([1, 3, 224, 224])
    conv1 = m.conv1
    bn1 = m.bn1
    o_output = bn1(conv1(p))
    fusion = fuse(conv1, bn1)
    f_output = fusion(p)
    print(o_output[0][0][0][0].item())
    print(f_output[0][0][0][0].item())
    print("Max abs diff: ", (o_output - f_output).abs().max().item())

print("===============")
print("module level test") 
m = torchvision.models.resnet18(True)
m.eval()
test_net(m)

print("============================")
print("Layer level test: ")
m = torchvision.models.resnet18(True)
m.eval()
test_layer(m)