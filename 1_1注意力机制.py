import torch
import torch.nn as nn

class selfAttention(nn.Module):
    """缩放点注意力机制, self-attention自注意力机制， scaled dot-product attention"""
    def __init__(self, scale):
        super().__init__()

        self.scale = scale                      # 缩放因子, 防止点积过大，缩放因子为根号下d_k, d_k为Q和K的维度  
        self.softmax = nn.Softmax(dim=-1)        # 在最后一个维度上进行softmax，计算自注意力权重的激活函数

    def forward(self, q, k, v, mask=None):
        u = torch.matmul(q, k.transpose(1,2))      # 计算Q和K的点积
        u = u / self.scale                      # 缩放

        if mask is not None:
            u = u.masked_fill(mask, -1e9)       # 将mask为True的位置替换为-1e9，这样在softmax后就会接近0,
        
        attn = self.softmax(u)                  # 计算注意力权重
        output = torch.bmm(attn, v)             # 计算最终输出

        return output, attn
    
if __name__ == "__main__":
    # 测试
    q = torch.randn(2, 4, 8)    # batch_size=2, 序列长度=4, d_model=8
    k = torch.randn(2, 4, 8)    # batch_size=2, 序列长度=4, d_model=8
    v = torch.randn(2, 4, 8)    # batch_size=2, 序列长度=4, d_model=8
    mask = torch.randn(2, 4, 4).random_(0, 2).bool()    # batch_size=2, 序列长度=4, 生成0-2的随机数，然后转换为bool类型

    scale = 8 ** 0.5
    attention = selfAttention(scale)
    output, attn = attention(q, k, v, mask)
    
    print(output)
    print(output.size())