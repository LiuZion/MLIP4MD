from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import DimeNetPlusPlus
import torch

dataset = QM9(root='./M1/DimeNet++/data/QM9') # 创建一个QM9数据集对象，指定数据存储路径

# 预测目标：Energy
target = 7

# 加载数据集并创建数据加载器, train_loader_shape = len(dataset)/batch_size, shuffle=True 表示打乱数据顺序
train_loader = DataLoader(dataset[:10000], batch_size=32, shuffle=True)

model = DimeNetPlusPlus(
    hidden_channels=128,
    out_channels=1,
    num_blocks=4,
    int_emb_size=64,
    basis_emb_size=8,
    out_emb_channels=256,
    num_spherical=7,
    num_radial=6,
    cutoff=5.0
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练循环
model.train()
max_steps = 100
for epoch in range(100):
    step = 0
    for data in train_loader:
        """
        data = data.to(device) 这行代码的作用是将数据（在这里是分子图数据）移动到指定的计算设备上（如GPU或CPU）。具体解释如下：
            1. data 是一个PyTorch Geometric的 Data 对象，包含分子的节点特征、位置信息等
            2. .to(device) 是PyTorch的方法，
                - 如果 device=cuda（即有可用GPU），则将数据复制到GPU显存
                - 如果 device=cpu，则数据保留在内存中
        """
        data = data.to(device)

        # 前向传播，计算输出和损失函数值
        out = model(data.z, data.pos, data.batch) # data.batch的取值范围是[0, batch_size-1], 第i个图的所有节点在batch_zise中都标记为i
        loss = (out.squeeze(-1) - data.y[:, target]).abs().mean()
        
        # 反向传播，更新模型参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        step += 1
        if step % 5 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.6f}")
        if step >= max_steps:
            break
