import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 简单的全连接网络模块
def build_net():
    return nn.Sequential(
        nn.Linear(2, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1)
    )

# 主模型：激波左侧和右侧两个网络 + 激波位置
class ShockPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_L = build_net()
        self.net_R = build_net()
        # 初始激波位置
        self.shock_pos_init = nn.Parameter(torch.tensor([0.0], requires_grad=True))
        # 激波移动速度
        self.shock_speed = nn.Parameter(torch.tensor([0.5], requires_grad=True))
        
    def shock_position(self, t):
        # 激波位置随时间变化: s(t) = s0 + v*t
        return self.shock_pos_init + self.shock_speed * t

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        # 计算当前时间的激波位置
        shock_pos = self.shock_position(t)
        left_mask = (x < shock_pos).float()
        right_mask = 1.0 - left_mask
        u_L = self.net_L(xt)
        u_R = self.net_R(xt)
        return left_mask * u_L + right_mask * u_R, u_L, u_R, shock_pos

# 初始化模型
model = ShockPINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 初始条件
def u0(x):
    return torch.where(x < 0.0, torch.ones_like(x), torch.zeros_like(x))

# 采样训练点
N_f = 2000
N_ic = 200
x_f = torch.rand(N_f, 1, device=device)*2-1
x_f.requires_grad = True

t_f = torch.rand(N_f, 1, device=device, requires_grad=True)

x_ic = torch.linspace(-1, 1, N_ic).view(-1, 1).to(device)
t_ic = torch.zeros_like(x_ic)

# 训练过程
def train(epochs):
    for epoch in range(epochs):
        optimizer.zero_grad()

        # 前向传播计算模型输出和当前激波位置
        u, u_L, u_R, shock_pos_f = model(x_f, t_f)

        # 求导计算 PDE 残差（在左右子域分别计算）
        u_L_t = torch.autograd.grad(u_L, t_f, grad_outputs=torch.ones_like(u_L), create_graph=True)[0]
        u_L_x = torch.autograd.grad(u_L, x_f, grad_outputs=torch.ones_like(u_L), create_graph=True)[0]
        f_L = (x_f < shock_pos_f).float() * (u_L_t + u_L * u_L_x)

        u_R_t = torch.autograd.grad(u_R, t_f, grad_outputs=torch.ones_like(u_R), create_graph=True)[0]
        u_R_x = torch.autograd.grad(u_R, x_f, grad_outputs=torch.ones_like(u_R), create_graph=True)[0]
        f_R = (x_f >= shock_pos_f).float() * (u_R_t + u_R * u_R_x)

        L_pde = torch.mean(f_L**2 + f_R**2)

        # 初始条件损失（整个域）
        u_ic, _, _, _ = model(x_ic, t_ic)
        L_ic = torch.mean((u_ic - u0(x_ic))**2)
        # 边界条件损失
        x_bc_left = torch.zeros(100, 1, device=device) - 1.0  # 左边界 x=-1
        x_bc_right = torch.zeros(100, 1, device=device) + 1.0  # 右边界 x=1
        t_bc = torch.rand(100, 1, device=device)  # 随机时间点
        
        # 计算边界预测值
        u_bc_left, _, _, _ = model(x_bc_left, t_bc)
        u_bc_right, _, _, _ = model(x_bc_right, t_bc)
        
        # 边界条件: u(-1,t) = 1, u(1,t) = 0
        L_bc = torch.mean((u_bc_left - 1.0)**2 + u_bc_right**2)
        # Rankine-Hugoniot 条件（激波处）
        # 在 t=0.5 处构造 RH 损失
        t_rh = torch.rand(1, 1, device=device).requires_grad_(True)
        
        # 计算t_rh时刻的激波位置
        shock_pos_rh = model.shock_position(t_rh)
        x_rh = shock_pos_rh.detach()  # 分离，避免梯度干扰
        
        # 预测激波处的左右值
        _, uL_rh, uR_rh, _ = model(x_rh, t_rh)
        
        # 计算激波位置对时间的导数
        s_t = torch.autograd.grad(shock_pos_rh, t_rh, grad_outputs=torch.ones_like(shock_pos_rh), create_graph=True)[0]
        
        # Rankine-Hugoniot条件: s'(t) = (f(uR) - f(uL))/(uR - uL)
        # 对于Burgers方程: f(u) = 0.5*u^2
        rh_res = s_t - (0.5 * uR_rh**2 - 0.5 * uL_rh**2) / (uR_rh - uL_rh + 1e-6)
        L_rh = torch.mean(rh_res**2)

        # 总损失
        loss = L_pde + L_ic + L_rh + L_bc
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            with torch.no_grad():
                current_shock_pos = model.shock_position(torch.tensor([[0.5]]).to(device)).item()
                shock_speed = model.shock_speed.item()
            print(f"Epoch {epoch}: Loss={loss.item():.2e}, PDE={L_pde.item():.2e}, IC={L_ic.item():.2e}, RH={L_rh.item():.2e}")
            print(f"Shock pos at t=0.5: {current_shock_pos:.3f}, Shock speed: {shock_speed:.3f}")

# 训练模型
train(2500)

# 预测结果可视化
def plot_result(t_eval=0.5):
    x_plot = torch.linspace(-1, 1, 500).view(-1, 1).to(device)
    t_plot = torch.full_like(x_plot, t_eval)
    with torch.no_grad():
        u_pred, _, _, shock_pos = model(x_plot, t_plot)
        shock_pos_value = shock_pos[0].item()  # 获取标量值
    
    u_exact = u0(x_plot-0.25)
    print(torch.mean(torch.abs(u_exact-u_pred)))
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot.cpu(), u_pred.cpu(), 'o', label="ShockPINN")
    plt.plot(x_plot.cpu(), u_exact.cpu(), label="Exact")
    plt.axvline(shock_pos_value, color='red', linestyle='--', label=f"Shock at t={t_eval}")
    plt.title(f"Burgers Equation at t={t_eval}")
    plt.xlabel("x"); plt.ylabel("u(x,t)")
    plt.legend(); plt.grid(True)
    plt.savefig(f"burgers_shock_t{t_eval}.png")
    plt.show()

# 绘制不同时间点的解
#plot_result(0.0)
plot_result(0.5)
#plot_result(1.0)
