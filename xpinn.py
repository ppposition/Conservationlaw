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

class SubNet(nn.Module):
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

class XPINN(nn.Module):
    def __init__(self, n_domains=2, in_dim=2, fourier_dim=100, scales=None):
        super().__init__()
        
        if scales is None:
            scales = [1.0] * n_domains
        
        self.subnets = nn.ModuleList([
            SubNet(in_dim, fourier_dim, scale) for scale in scales
        ])
        
        self.interfaces = torch.linspace(-1, 1, n_domains+1)[1:-1]
        
    def forward(self, x, t):
        batch_size = x.shape[0]
        result = torch.zeros(batch_size, 1).to(x.device)
        
        masks = []
        prev_interface = -float('inf')
        
        for i, interface in enumerate(self.interfaces):
            mask = (x >= prev_interface) & (x < interface)
            masks.append(mask)
            prev_interface = interface
        
        masks.append((x >= prev_interface))
        
        for i, subnet in enumerate(self.subnets):
            mask = masks[i]
            if torch.any(mask):
                x_subnet = x[mask].reshape(-1, 1)
                t_subnet = t[mask].reshape(-1, 1)
                result[mask] = subnet(x_subnet, t_subnet).squeeze(-1)
        
        return result
    
    def get_interface_values(self, t_points):
        interface_values = []
        
        for interface in self.interfaces:
            x_left = torch.ones_like(t_points) * (interface - 1e-6)
            x_right = torch.ones_like(t_points) * (interface + 1e-6)
            
            u_left = self.forward(x_left, t_points)
            u_right = self.forward(x_right, t_points)
            
            interface_values.append((u_left, u_right))
            
        return interface_values
    
    def get_interface_derivatives(self, t_points):
        interface_derivatives = []
        
        for interface in self.interfaces:
            x_left = torch.ones_like(t_points) * (interface - 1e-6)
            x_left.requires_grad = True
            u_left = self.forward(x_left, t_points)
            
            x_right = torch.ones_like(t_points) * (interface + 1e-6)
            x_right.requires_grad = True
            u_right = self.forward(x_right, t_points)
            
            u_x_left = torch.autograd.grad(
                u_left, x_left, 
                grad_outputs=torch.ones_like(u_left),
                create_graph=True
            )[0]
            
            u_x_right = torch.autograd.grad(
                u_right, x_right, 
                grad_outputs=torch.ones_like(u_right),
                create_graph=True
            )[0]
            
            interface_derivatives.append((u_x_left, u_x_right))
            
        return interface_derivatives

def solve_burgers_with_xpinn(u0_func, u_true_func, l, r, T, 
                            n_domains=2, scales=None, 
                            N_f=10000, N_f_per_domain=None,
                            N_i=100, N_b=100, N_interface=50,
                            epochs=5000, boundary='Dirichlet',
                            b_l=0, b_r=0, draw=False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    
    model = XPINN(n_domains=n_domains, scales=scales).to(device)
    
    if N_f_per_domain is None:
        N_f_per_domain = N_f // n_domains
    
    x_i = torch.linspace(l, r, N_i).reshape(-1,1).to(device)
    t_i = torch.zeros_like(x_i).to(device)
    
    x_b_l = torch.ones(N_b, 1).to(device) * l
    x_b_r = torch.ones(N_b, 1).to(device) * r
    t_b = torch.linspace(0, T, N_b).reshape(-1,1).to(device)
    
    t_interface = torch.linspace(0, T, N_interface).reshape(-1,1).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    flux = lambda u: 0.5 * u**2
    
    errors = []
    losses = []
    
    for epoch in range(epochs+1):
        optimizer.zero_grad()
        
        domain_losses = []
        
        prev_interface = l
        
        for i in range(n_domains):
            if i < n_domains - 1:
                next_interface = model.interfaces[i].item()
            else:
                next_interface = r
            
            x_f_domain = torch.rand(N_f_per_domain, 1, device=device) * (next_interface - prev_interface) + prev_interface
            x_f_domain.requires_grad = True
            t_f_domain = torch.rand(N_f_per_domain, 1, device=device) * T
            t_f_domain.requires_grad = True
            
            u = model(x_f_domain, t_f_domain)
            u_x = torch.autograd.grad(u, x_f_domain, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            f_u_x = torch.autograd.grad(flux(u), x_f_domain, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            u_t = torch.autograd.grad(u, t_f_domain, grad_outputs=torch.ones_like(u), create_graph=True)[0]
            
            pde_residual = u_t + f_u_x
            
            domain_loss = torch.mean(pde_residual**2)
            domain_losses.append(domain_loss)
            
            prev_interface = next_interface
        
        loss_f = sum(domain_losses) / n_domains
        
        u_i_pred = model(x_i, t_i)
        u_i_true = u0_func(x_i)
        loss_i = nn.MSELoss()(u_i_pred, u_i_true)
        
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
        
        interface_values = model.get_interface_values(t_interface)
        loss_interface_values = 0
        for u_left, u_right in interface_values:
            loss_interface_values += nn.MSELoss()(u_left, u_right)
        
        interface_derivatives = model.get_interface_derivatives(t_interface)
        loss_interface_derivatives = 0
        for u_x_left, u_x_right in interface_derivatives:
            loss_interface_derivatives += nn.MSELoss()(u_x_left, u_x_right)
        
        loss = loss_f + loss_i + loss_b + loss_interface_values + 0.1 * loss_interface_derivatives
        
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 500 == 0:
            x_plot = torch.linspace(l, r, 200).reshape(-1, 1).to(device)
            t_plot = torch.full_like(x_plot, T).to(device)
            
            u_pred = model(x_plot, t_plot).detach().cpu().numpy()
            u_true = u_true_func(x_plot.cpu().numpy(), T)
            error = np.mean(np.abs(u_pred - u_true))    
            print(f"Epoch {epoch}, Loss: {loss.item():.5e}, Error: {error:.2e}")
            print(f"  PDE: {loss_f.item():.2e}, IC: {loss_i.item():.2e}, BC: {loss_b.item():.2e}")
            print(f"  Interface Value: {loss_interface_values.item():.2e}, Deriv: {loss_interface_derivatives.item():.2e}")
            errors.append(error)
    
    end_time = time.time()
    print(f"Training time: {end_time-start_time} seconds")
    
    x_plot = torch.linspace(l, r, 200).reshape(-1, 1).to(device)
    t_plot = torch.full_like(x_plot, T).to(device)
    
    u_pred = model(x_plot, t_plot).detach().cpu().numpy()
    u_true = u_true_func(x_plot.cpu().numpy(), T)
    
    final_error = np.mean(np.abs(u_pred - u_true))
    print(f"Final error: {final_error:.2e}")
    
    if draw:
        plt.figure(figsize=(10, 6))
        plt.plot(x_plot.cpu(), u_pred, label="XPINN prediction")
        plt.plot(x_plot.cpu(), u_true, '--', label="Analytical solution")
        
        for interface in model.interfaces:
            plt.axvline(interface.item(), color='gray', linestyle='-', alpha=0.5)
            
        plt.xlabel("x")
        plt.ylabel("u(x,t)")
        plt.title(f"XPINN for Burgers equation (t={T})")
        plt.legend()
        plt.grid(True)
        
        if not os.path.exists('xpinn_results'):
            os.makedirs('xpinn_results')
        plt.savefig(f'xpinn_results/xpinn_burgers_{n_domains}domains_shock.png')
    
    return model, final_error

def main():
    if not os.path.exists('xpinn_results'):
        os.makedirs('xpinn_results')
    
    l = -1
    r = 1   
    T = 0.5
    
    u0_func = lambda x: torch.where(x<0.0, 1.0, 0.0)
    u_true = lambda x, t: np.where(x<t/2, 1.0, 0.0)
    
    model, error = solve_burgers_with_xpinn(
        u0_func=u0_func,
        u_true_func=u_true,
        l=l,
        r=r,
        T=T,
        n_domains=3,
        scales=[1.0, 5.0, 1.0],
        N_f=15000,
        N_i=200,
        N_b=100,
        N_interface=100,
        epochs=10000,
        boundary='Dirichlet',
        b_l=1.0,
        b_r=0.0,
        draw=True
    )
    print(f"Shock wave error with XPINN = {error:.4e}")
    
    u0_func_rarefaction = lambda x: torch.where(x<0.0, 0.0, 1.0)
    u_true_rarefaction = lambda x, t: np.where(x<0.0, 0.0, np.where(x<t, x/t, 1.0))
    
    model_rare, error_rare = solve_burgers_with_xpinn(
        u0_func=u0_func_rarefaction,
        u_true_func=u_true_rarefaction,
        l=l,
        r=r,
        T=T,
        n_domains=3,
        scales=[1.0, 4.0, 1.0],
        N_f=15000,
        N_i=200,
        N_b=100,
        N_interface=100,
        epochs=10000,
        boundary='Dirichlet',
        b_l=0.0,
        b_r=1.0,
        draw=True
    )
    print(f"Rarefaction wave error with XPINN = {error_rare:.4e}")
    
    return {
        "shock_model": model,
        "shock_error": error,
        "rarefaction_model": model_rare,
        "rarefaction_error": error_rare
    }

if __name__ == "__main__":
    results = main()