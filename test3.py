import torch

# 创建一个具有大小为1的单维度的张量
tensor = torch.tensor([[1,2,3,4]])
print(tensor)
# 使用squeeze函数移除大小为1的维度
squeezed_tensor = torch.squeeze(tensor, dim=0)
print(squeezed_tensor.shape)
print(squeezed_tensor)