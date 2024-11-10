import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim_model, num_heads, dropout=0.1):
        super().__init__()

        self.dim_model = dim_model                  # 模型的维度
        self.num_heads = num_heads                  # 多头注意力的头数
        
        self.linear_q = nn.Linear(dim_model, dim_model)
        self.linear_k = nn.Linear(dim_model, dim_model)
        self.linear_v = nn.Linear(dim_model, dim_model)
        self.linear_out = nn.Linear(dim_model, dim_model)
        self._softmax = nn.Softmax(dim=-2)

    def forward(self, q, k, v):
            batch_size, seq_len, dim_model = q.size()
            head_dim = self.dim_model // self.num_heads   # 每个头的维度

            q = self.linear_q(q).view(batch_size, seq_len, self.num_heads, head_dim).permute(0, 2, 1, 3)
            k = self.linear_k(k).view(batch_size, seq_len, self.num_heads, head_dim).permute(0, 2, 1, 3)
            v = self.linear_v(v).view(batch_size, seq_len, self.num_heads, head_dim).permute(0, 2, 1, 3)

            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            mask = torch.triu(torch.ones(seq_len, seq_len, dtype=bool)) # 生成上三角矩阵, 用于mask掉未来信息
            scores = scores.masked_fill(mask, -1e9)
            scores = torch.matmul(self._softmax(scores), v)

            score = scores.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, self.dim_model)
            output = self.linear_out(score)

            return output
        
if __name__ == "__main__":
    q = torch.randn(2, 4, 8)    # batch_size=2, 序列长度=4, d_model=8
    k = torch.randn(2, 4, 8)    # batch_size=2, 序列长度=4, d_model=8
    v = torch.randn(2, 4, 8)    # batch_size=2, 序列长度=4, d_model=8

    attention = MultiHeadSelfAttention(dim_model=8, num_heads=2)
    output = attention(q, k, v)
    print(output.shape)