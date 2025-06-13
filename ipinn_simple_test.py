import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleIPINN(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), 
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.SELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)
    
    def get_solution(self, x, t):
        x.requires_grad_(True)
        integral = self.forward(x, t)
        u = torch.autograd.grad(
            integral, x, 
            grad_outputs=torch.ones_like(integral),
            create_graph=True
        )[0]
        return u

def test_shock_problem():
    print("Testing IPINN solving shock problem...")
    
    l, r, T = -1.0, 1.0, 0.5
    
    model = SimpleIPINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    
    flux = lambda u: 0.5 * u**2
    
    u0_func = lambda x: torch.where(x < 0.0, 1.0, 0.0)
    u_true_func = lambda x, t: np.where(x < t/2, 1.0, 0.0)
    
    N_f, N_i, N_b = 10000, 1000, 1000
    epochs = 6000
    
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        
        x_a = torch.rand(N_f, 1) * (r - l) + l
        x_b = torch.rand(N_f, 1) * (r - l) + l
        x_a, x_b = torch.min(x_a, x_b), torch.max(x_a, x_b)
        x_a, x_b = x_a.to(device), x_b.to(device)
        t_f = torch.rand(N_f, 1) * T
        t_f = t_f.to(device)
        t_f.requires_grad_(True)
        
        I_a = model(x_a, t_f)
        I_b = model(x_b, t_f)  
        I_diff = I_b - I_a
        
        dI_dt = torch.autograd.grad(
            I_diff, t_f,
            grad_outputs=torch.ones_like(I_diff),
            create_graph=True
        )[0]
        
        u_a = model.get_solution(x_a, t_f)
        u_b = model.get_solution(x_b, t_f)
        flux_diff = flux(u_a) - flux(u_b)
        
        loss_ipinn = nn.MSELoss()(dI_dt, flux_diff)
        
        x_i = torch.rand(N_i, 1) * (r - l) + l
        x_i = x_i.to(device)
        t_i = torch.zeros_like(x_i)
        x_b_l = torch.ones(N_b, 1) * l
        x_b_r = torch.ones(N_b, 1) * r
        t_b = torch.rand(N_b, 1) * T
        x_b_l, x_b_r, t_b = x_b_l.to(device), x_b_r.to(device), t_b.to(device)
        
        u_i_pred = model.get_solution(x_i, t_i)
        u_i_true = u0_func(x_i)
        loss_i = nn.MSELoss()(u_i_pred, u_i_true)
        
        u_l = model.get_solution(x_b_l, t_b)
        u_r = model.get_solution(x_b_r, t_b)
        loss_b = nn.MSELoss()(u_l, torch.ones_like(u_l)) + nn.MSELoss()(u_r, torch.zeros_like(u_r))
        
        I_left = model(x_b_l, t_b)
        loss_integral_bc = nn.MSELoss()(I_left, torch.zeros_like(I_left))
        
        loss = loss_ipinn + loss_i + 0.5*loss_b + loss_integral_bc
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        if epoch % 400 == 0:
            x_test = torch.linspace(l, r, 100).reshape(-1, 1)
            t_test = torch.full_like(x_test, T)
            x_test, t_test = x_test.to(device), t_test.to(device)
            u_pred = model.get_solution(x_test, t_test).detach().cpu().numpy()
            u_true = u_true_func(x_test.detach().cpu().numpy(), T)
            error = np.mean(np.abs(u_pred - u_true))
            print(f"Epoch {epoch}, Loss: {loss.item():.4e}, Error: {error:.4e}")
    
    x_plot = torch.linspace(l, r, 200).reshape(-1, 1)
    t_plot = torch.full_like(x_plot, T)
    x_plot, t_plot = x_plot.to(device), t_plot.to(device)
    u_pred = model.get_solution(x_plot, t_plot).detach().cpu().numpy()
    u_true = u_true_func(x_plot.detach().cpu().numpy(), T)
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot.detach().cpu().numpy(), u_pred, 'b-', label='IPINN prediction', linewidth=2)
    plt.plot(x_plot.detach().cpu().numpy(), u_true, 'r--', label='Analytical solution', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(f'IPINN solving Burgers equation shock problem (t={T})')
    plt.legend()
    plt.grid(True)
    plt.savefig('ipinn_simple_test.png')
    plt.show()
    
    final_error = np.mean(np.abs(u_pred - u_true))
    print(f"Final error: {final_error:.4e}")
    
    return final_error

def test_rarefaction_problem():
    print("Testing IPINN solving rarefaction problem...")
    
    l, r, T = -1.0, 1.0, 0.4
    
    model = SimpleIPINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    
    flux = lambda u: 0.5 * u**2
    
    u_l, u_r = -1.0, 1.0
    u0_func = lambda x: torch.where(x < 0.0, u_l, u_r)
    
    def u_true_func(x, t):
        u_true = np.zeros_like(x)
        mask_left = x <= u_l * t
        u_true[mask_left] = u_l
        
        mask_fan = (u_l * t < x) & (x < u_r * t)
        u_true[mask_fan] = x[mask_fan] / t if t > 0 else u_r
        
        mask_right = x >= u_r * t
        u_true[mask_right] = u_r
        
        return u_true
    
    N_f, N_i, N_b = 15000, 1500, 1000
    epochs = 8000
    
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        
        x_a = torch.rand(N_f, 1) * (r - l) + l
        x_b = torch.rand(N_f, 1) * (r - l) + l
        x_a, x_b = torch.min(x_a, x_b), torch.max(x_a, x_b)
        x_a, x_b = x_a.to(device), x_b.to(device)
        t_f = torch.rand(N_f, 1) * T
        t_f = t_f.to(device)
        t_f.requires_grad_(True)
        
        I_a = model(x_a, t_f)
        I_b = model(x_b, t_f)  
        I_diff = I_b - I_a
        
        dI_dt = torch.autograd.grad(
            I_diff, t_f,
            grad_outputs=torch.ones_like(I_diff),
            create_graph=True
        )[0]
        
        u_a = model.get_solution(x_a, t_f)
        u_b = model.get_solution(x_b, t_f)
        flux_diff = flux(u_a) - flux(u_b)
        
        loss_ipinn = nn.MSELoss()(dI_dt, flux_diff)
        
        x_i = torch.rand(N_i, 1) * (r - l) + l
        x_i = x_i.to(device)
        t_i = torch.zeros_like(x_i)
        
        x_b_l = torch.ones(N_b, 1) * l
        x_b_r = torch.ones(N_b, 1) * r
        t_b = torch.rand(N_b, 1) * T
        x_b_l, x_b_r, t_b = x_b_l.to(device), x_b_r.to(device), t_b.to(device)
        
        u_i_pred = model.get_solution(x_i, t_i)
        u_i_true = u0_func(x_i)
        loss_i = nn.MSELoss()(u_i_pred, u_i_true)
        
        u_l_pred = model.get_solution(x_b_l, t_b)
        u_r_pred = model.get_solution(x_b_r, t_b)
        loss_b = nn.MSELoss()(u_l_pred, torch.full_like(u_l_pred, u_l)) + \
                 nn.MSELoss()(u_r_pred, torch.full_like(u_r_pred, u_r))
        
        I_left = model(x_b_l, t_b)
        loss_integral_bc = nn.MSELoss()(I_left, torch.zeros_like(I_left))
        
        loss = loss_ipinn + 2.0*loss_i + 0.8*loss_b + loss_integral_bc
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        if epoch % 500 == 0:
            x_test = torch.linspace(l, r, 100).reshape(-1, 1)
            t_test = torch.full_like(x_test, T)
            x_test, t_test = x_test.to(device), t_test.to(device)
            u_pred = model.get_solution(x_test, t_test).detach().cpu().numpy()
            u_true = u_true_func(x_test.detach().cpu().numpy(), T)
            error = np.mean(np.abs(u_pred.flatten() - u_true.flatten()))
            print(f"Epoch {epoch}, Loss: {loss.item():.4e}, Error: {error:.4e}")
    
    x_plot = torch.linspace(l, r, 200).reshape(-1, 1)
    t_plot = torch.full_like(x_plot, T)
    x_plot, t_plot = x_plot.to(device), t_plot.to(device)
    u_pred = model.get_solution(x_plot, t_plot).detach().cpu().numpy()
    u_true = u_true_func(x_plot.detach().cpu().numpy(), T)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot.detach().cpu().numpy(), u_pred.flatten(), 'b-', label='IPINN prediction', linewidth=2)
    plt.plot(x_plot.detach().cpu().numpy(), u_true.flatten(), 'r--', label='Analytical solution', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title(f'IPINN solving Burgers equation rarefaction problem (t={T})')
    plt.legend()
    plt.grid(True)
    plt.savefig('ipinn_rarefaction_test.png')
    plt.show()
    
    final_error = np.mean(np.abs(u_pred.flatten() - u_true.flatten()))
    print(f"Final error for rarefaction problem: {final_error:.4e}")
    
    return final_error

if __name__ == "__main__":
    
    print("shock problem:")
    shock_error = test_shock_problem()
    print(f"shock problem error: {shock_error:.4e}")
    
    print("rarefaction problem:")
    rarefaction_error = test_rarefaction_problem()
    print(f"rarefaction problem error: {rarefaction_error:.4e}")