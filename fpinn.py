import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import os

class FourierFeatureLayer(nn.Module):
    def __init__(self, in_dim, mapping_size=100, scale=10.0):
        super().__init__()
        B = torch.randn((mapping_size, in_dim)) * scale
        self.register_buffer("B", B)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class fPINN(nn.Module):
    def __init__(self, in_dim=2, fourier_dim=100, scale=1.0):
        super().__init__()
        self.ff = FourierFeatureLayer(in_dim, fourier_dim, scale=scale)
        self.net = nn.Sequential(
            nn.Linear(2 * fourier_dim, 128), 
            nn.SiLU(),
            nn.Linear(128, 128), 
            nn.SiLU(),
            nn.Linear(128, 128), 
            nn.SiLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        ff_xt = self.ff(xt)
        return self.net(ff_xt)

# 设置设备
def solve_pde_with_fpinn(u0_func, u_true_func, l, r, T, f, scale=1.0, N_f=10000, N_i=100, N_b=100, epochs=5000, boundary='periodic', b_l=0, b_r=0, draw=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    # 神经网络结构
    x_plot = torch.linspace(l, r, 200).reshape(-1, 1).to(device)
    t_plot = torch.full_like(x_plot, T).to(device)
    
    # 初始化模型
    model = fPINN(scale=scale).to(device)

    x_i = torch.linspace(l, r, N_i).reshape(-1,1).to(device)
    t_i = torch.zeros_like(x_i).to(device)

    x_b_l = torch.ones(N_b, 1).to(device) * l
    x_b_r = torch.ones(N_b, 1).to(device) * r
    t_b = torch.linspace(0, T, N_b).reshape(-1,1).to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
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

        # 总损失
        loss = loss_f + loss_i + loss_b
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 500 == 0:
            u_pred = model(x_plot, t_plot).detach().cpu().numpy()
            u_true = u_true_func(x_plot.cpu().numpy(), T)
            error = np.mean(np.abs(u_pred - u_true))    
            print(f"Epoch {epoch}, Loss: {loss.item():.5e}, mean residual = {error:.2e}")
            errors.append(error)
    end_time = time.time()
    print(f"time for PINN: {end_time-start_time} seconds")
    u_pred = model(x_plot, t_plot).detach().cpu().numpy()
    u_true = u_true_func(x_plot.cpu().numpy(), T)

    error = np.mean(np.abs(u_pred - u_true))
    print(f"mean residual = {error:.2e}")
    if draw:    
        plt.figure(figsize=(10, 6))
        plt.plot(x_plot.cpu(), u_pred, label="PINN prediction")
        plt.plot(x_plot.cpu(), u_true, '--', label="Analytical solution")
        plt.xlabel("x")
        plt.ylabel("u(x,t)")
        plt.title(f"fPINN for linear continuous conservation law")
        plt.legend()
        plt.grid(True)
        plt.savefig(f'fpinn_results/fpinn_linear_continuous_{scale}.png')
    return model, error

def main():
    if not os.path.exists('fpinn_results'):
        os.makedirs('fpinn_results')
    l = -1  
    r = 1
    T = 0.5
    u0_func = lambda x: torch.where(x<0.0, 1.0, 0.0)
    u_true = lambda x, t: np.where(x<t/2, 1.0, 0.0)
    f = lambda u: u**2/2
    
    #min_error = float('inf')
    model, error = solve_pde_with_fpinn(u0_func, u_true, l, r, T, f, scale=1.0, N_f=10000, N_i=100, N_b=100, epochs=10000, boundary='Dirichlet', b_l=1.0, b_r=0.0, draw=True)
    print(f"shock wave error = {error:.4e}")
    
    '''u_0 = lambda x: torch.where(x<0.0, 0.0, 1.0)
    u_true = lambda x, t: np.where(x<0.0, 0.0, np.where(x<t, x/t, 1.0))
    min_error = float('inf')
    for scale in np.linspace(0.1, 5.0, 10):
        print(f"scale = {scale}")
        model, error = solve_pde_with_fpinn(u_0, u_true, l, r, T, f, scale=scale, N_f=10000, N_i=100, N_b=100, epochs=10000, boundary='Dirichlet', b_l=0.0, b_r=1.0, draw=True)
        if error < min_error:
            min_error = error
            best_model = model
    print(f"best scale = {scale}, error = {min_error}")
    return best_model'''

if __name__ == "__main__":
    main()