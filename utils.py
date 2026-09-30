import torch
import os
import numpy as np

device='cuda' if torch.cuda.is_available() else 'cpu'
device='mps' if torch.backends.mps.is_available() else device

def D_train(x, G, D, D_optimizer, criterion):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.to(device), y_real.to(device)

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on facke
    z = torch.randn(x.shape[0], 100).to(device)
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).to(device)

    D_output =  D(x_fake)
    
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()
        
    return  D_loss.data.item()


def G_train(x, G, D, G_optimizer, criterion):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).to(device)
    y = torch.ones(x.shape[0], 1).to(device)
                 
    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
        
    return G_loss.data.item()



def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G, folder):
    ckpt = torch.load(os.path.join(folder,'G.pth'),map_location='cpu')
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G

#DRS
def load_discriminator(D, folder):
    ckpt = torch.load(os.path.join(folder, 'D.pth'), map_location='cpu')
    D.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return D

def discriminator_rejection_sampling(G, D, n_samples, batch_size=2048, eps = 1e-6, gamma = -5.0):
    G.eval()
    D.eval()

    accepted_samples = []
    accepted_count = 0
    M = -float("inf") #estimate of max logit
    with torch.no_grad():
        while accepted_count < n_samples:
            z = torch.randn(batch_size, 100).to(device)
            x = G(z)

            Dx = D(x).view(-1)
            Dx = torch.clamp(Dx, eps, 1 - eps) # because Dx is a sigmoid and it can be 0 or 1 which is not good for us
            
            logit = torch.log(Dx) - torch.log(1 - Dx) # from sigmoid to logit

            batch_max = logit.max().item()
            if batch_max > M:
                M = batch_max
            

            delta = logit - M - eps
            delta = torch.clamp(delta, max=-eps)  # garantit exp(delta) < 1

            F_x = logit - M - torch.log1p(- torch.exp(delta)) - gamma
            accept_prob = torch.sigmoid(F_x)
            # accept_prob = 1/(1 + torch.exp(-F_x))
            
        
            u = torch.rand_like(accept_prob) # on met accept_prob dans argument pour pouvoir comparer les deux
            keep = u < accept_prob
            accept_rate = keep.float().mean().item()
            if keep.any():
                kept = x[keep]
                accepted_samples.append(kept)
                accepted_count += kept.shape[0]
            
            print(f"Accepted rate {accept_rate*100:.3f}% | M : {M} | Total: {accepted_count}/{n_samples} | batch accept rate: {accept_prob.mean().item()*100:.2f}%")
            
    out = torch.cat(accepted_samples, dim=0)
    N = out.size(0)
    idx = torch.randperm(N, device=out.device)[:n_samples]
    selected_samples = out[idx]
    return selected_samples



def estimate_r_and_cK(G, D, K, sample_size=10000, eps=1e-6):
    G.eval()
    D.eval()
    r_values = []
    with torch.no_grad():
        for _ in range(sample_size // 2048 + 1):
            z = torch.randn(2048, 100).to(device)
            x = G(z)
            Dx = D(x).view(-1)
            Dx = torch.clamp(Dx, eps, 1 - eps)
            logit = torch.log(Dx) - torch.log(1 - Dx)
            r = torch.exp(logit)  # r(x) = exp(logit) = D/(1-D)
            r_values.append(r.cpu())
    r_values = torch.cat(r_values).numpy()
    
    # Recherche dichotomique pour c_K afin de respecter budget K
    low, high = 1e-6, r_values.max()
    for _ in range(30):
        mid = 0.5 * (low + high)
        accept_rates = np.minimum(r_values / mid, 1.0)
        if accept_rates.mean() < 1/K:
            high = mid
        else:
            low = mid
    c_K = low
    return c_K

def optimal_budgeted_rejection_sampling(G, D, n_samples, c_K, batch_size=2048, eps=1e-6):
    G.eval()
    D.eval()
    accepted_samples = []
    accepted_count = 0

    with torch.no_grad():
        while accepted_count < n_samples:
            z = torch.randn(batch_size, 100).to(device)
            x = G(z)
            Dx = D(x).view(-1)
            Dx = torch.clamp(Dx, eps, 1 - eps)
            logit = torch.log(Dx) - torch.log(1 - Dx)
            r = torch.exp(logit)
            
            accept_prob = torch.clamp(r / c_K, max=1.0)
            
            u = torch.rand_like(accept_prob)
            keep = u < accept_prob
            if keep.any():
                kept = x[keep]
                accepted_samples.append(kept)
                accepted_count += kept.shape[0]
            
            print(f"Accepted count: {accepted_count}/{n_samples} | batch accept rate: {accept_prob.mean().item()*100:.2f}%")
    
    out = torch.cat(accepted_samples, dim=0)
    N = out.size(0)
    idx = torch.randperm(N, device=out.device)[:n_samples]
    selected_samples = out[idx]
    return selected_samples

