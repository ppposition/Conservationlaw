import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

# 设置设备
def solve_pde_with_lbpinn(u0_func, u_true_func, l, r, T, f, N_f=10000, N_i=100, N_b=100, epochs=5000, boundary='periodic', b_l=0, b_r=0, draw=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 神经网络结构
    class PINN(nn.Module):
        def __init__(self):
            super(PINN, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 64),
                nn.SiLU(),
                nn.Linear(64, 64),
                nn.SiLU(),
                nn.Linear(64, 64),
                nn.SiLU(),
                nn.Linear(64, 1)
            )

        def forward(self, x, t):
            xt = torch.cat((x, t), dim=1)
            return self.net(xt)

    # 初始化模型
    model = PINN().to(device)
    
    
    start_time = time.time()
    x_plot = torch.linspace(l, r, 200).reshape(-1, 1).to(device)
    t_plot = torch.full_like(x_plot, T).to(device)
    x_i = torch.linspace(l, r, N_i).reshape(-1,1).to(device)
    t_i = torch.zeros_like(x_i).to(device)

    x_b_l = torch.ones(N_b, 1).to(device) * l
    x_b_r = torch.ones(N_b, 1).to(device) * r
    t_b = torch.linspace(0, T, N_b).reshape(-1,1).to(device)

    log_sigma_f = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True, device=device))
    log_sigma_i = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True, device=device))
    log_sigma_b = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True, device=device))
    log_sigma_entropy = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True, device=device))
    # 优化器
    optimizer = torch.optim.Adam(list(model.parameters()) + [log_sigma_f, log_sigma_i, log_sigma_b, log_sigma_entropy], lr=1e-3)

    # 训练
    errors = []
    losses = []
    for epoch in range(epochs+1):
        # 构造训练点
        x_f = torch.rand(N_f, 1, device=device) * (r-l) + l
        x_f.requires_grad = True
        t_f = torch.rand(N_f, 1, device=device) * T * 1.1
        t_f.requires_grad = True
        optimizer.zero_grad()

        # PDE 残差
        u = model(x_f, t_f)
        f_u_x = torch.autograd.grad(f(u), x_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        f_u_t = u_t + f_u_x

        # 初始条件损失
        u_i_pred = model(x_i, t_i)
        u_i_true = u0_func(x_i)
        loss_i = nn.MSELoss()(u_i_pred, u_i_true)

        # 周期边界损失
        if boundary == 'periodic':
            u_l = model(x_b_l, t_b)
            u_r = model(x_b_r, t_b)
            loss_b = nn.MSELoss()(u_l, u_r)
        elif boundary == 'Dirichlet':
            u_l = model(x_b_l, t_b)
            u_r = model(x_b_r, t_b)
            if isinstance(b_l, float):
                b_l = torch.ones_like(u_l) * b_l
            if isinstance(b_r, float):
                b_r = torch.ones_like(u_r) * b_r
            loss_b = nn.MSELoss()(u_l, b_l) + nn.MSELoss()(u_r, b_r)
        # PDE残差损失
        loss_f = torch.mean(f_u_t**2)

        eta = lambda u: u**2/2
        phi = lambda u: u**3/3
        # 熵损失
        u_entropy = torch.autograd.grad(eta(u), t_f, grad_outputs=torch.ones_like(u), create_graph=True)[0] + torch.autograd.grad(phi(u), x_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        loss_entropy = torch.mean(torch.relu(u_entropy)**2)
        # 总损失
        loss = 1/(2*torch.exp(2*log_sigma_f)) * loss_f + 1/(2*torch.exp(2*log_sigma_i)) * loss_i + 1/(2*torch.exp(2*log_sigma_b)) * loss_b + 1/(2*torch.exp(2*log_sigma_entropy)) * loss_entropy + log_sigma_f + log_sigma_i + log_sigma_b + log_sigma_entropy
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            u_pred = model(x_plot, t_plot).detach().cpu().numpy()
            u_true = u_true_func(x_plot.cpu().numpy(), T)
            error = np.mean(np.abs(u_pred - u_true))    
            print(f"Epoch {epoch}, Loss: {loss.item():.5e}, mean residual = {error:.2e}")
            errors.append(error)
    end_time = time.time()
    print(f"time for LBPINN: {end_time-start_time} seconds")

    # 计算并绘制结果
    t_fixed = T
    u_pred = model(x_plot, t_plot).detach().cpu().numpy()
    u_true = u_true_func(x_plot.cpu().numpy(), t_fixed)

    error = np.mean(np.abs(u_pred - u_true))
    print(f"mean residual = {error:.2e}")
    if draw:    
        plt.figure(figsize=(10, 6))
        plt.plot(x_plot.cpu(), u_pred, label="PINN prediction")
        plt.plot(x_plot.cpu(), u_true, '--', label="Analytical solution")
        plt.xlabel("x")
        plt.ylabel("u(x,t)")
        plt.title(f"LBPINN for linear continuous conservation law")
        plt.legend()
        plt.grid(True)
        plt.savefig('lbpinn_linear_continuous.png')
        plt.show()
    
    return model, errors


l = -1  
r = 1
T = 0.5
u0_func = lambda x: torch.where(x<0.0, 1.0, 0.0)
u_true = lambda x, t: np.where(x<t/2, 1.0, 0.0)
model_pinn, errors_pinn = solve_pde_with_lbpinn(u0_func, u_true, l, r, T, f=lambda u: u**2/2, boundary='Dirichlet', b_l=1.0, b_r=0.0, epochs=5000, draw=True)