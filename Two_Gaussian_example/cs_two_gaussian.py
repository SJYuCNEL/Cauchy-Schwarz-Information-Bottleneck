# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 22:22:24 2023

@author: xyu1
"""

import matplotlib.pyplot as plt
import torch

from tqdm import tqdm


plt.rcParams['text.usetex'] = True


class TrainableData(torch.nn.Module):
    def __init__(self, n_data, dim):
        super().__init__()
        self.data = torch.nn.parameter.Parameter(torch.randn((n_data, dim)), requires_grad=True)

    def forward(self):
        return self.data


class CauchySchwarzLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_kernel, target_kernel, cross_kernel):
        return self._compute_cs(target_kernel, input_kernel, cross_kernel)

    def _compute_cs(self, K1, K2, K12):
        return -2 * torch.log(torch.sum(K12)) + torch.log(torch.sum(K1)) + torch.log(torch.sum(K2))


def compute_mmd(K1, K2, K12):
    N1 = K1.size(0)
    N2 = K2.size(0)

    return torch.sum(K1) / N1 ** 2 + torch.sum(K2) / N2 ** 2 - 2 * torch.sum(K12) / (N1 * N2)

def compute_kernel(X1, X2, sigma):
    sqnormX1 = (X1 ** 2).sum(dim=-1, keepdim=True)
    sqnormX2 = (X2 ** 2).sum(dim=-1, keepdim=True)
    inner = X1 @ X2.T

    return torch.exp( - 1. / (2 * sigma ** 2) * (sqnormX1 + sqnormX2.T - 2 * inner))

def get_device():
    return (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )

def init_data(n_data, dim, device, gt_n_data, gt_sigma, gt_mean_1, gt_mean_2):
    print(f"Initialize data with {n_data} datapoints in {dim} dimensions.")
    data_new = TrainableData(n_data=n_data, dim=dim).to(device)
    data_gaussain_1 = (torch.randn((gt_n_data, dim)) * gt_sigma + torch.Tensor(gt_mean_1)).to(device)
    data_gaussain_2 = (torch.randn((gt_n_data, dim)) * gt_sigma + torch.Tensor(gt_mean_2)).to(device)
    
    
    data_ground_truth = torch.cat((data_gaussain_1,data_gaussain_2),0)
    

    return data_new, data_ground_truth

def train(data_ground_truth, data_new, kernel_sigma, lr, epochs):
    K11 = compute_kernel(data_ground_truth, data_ground_truth, kernel_sigma)

    opt = torch.optim.SGD(data_new.parameters(), lr=lr)
    loss_fn = CauchySchwarzLoss()

    cs_loss = []
    mmd_loss = []

    data = data_new()
    for _ in tqdm(range(epochs)):
        K22 = compute_kernel(data, data, kernel_sigma)
        K12 = compute_kernel(data_ground_truth, data, kernel_sigma)
        loss = loss_fn(K22, K11, K12)
        cs_loss.append(loss.cpu().detach().numpy())
        mmd_loss.append(compute_mmd(K11, K22, K12).cpu().detach().numpy())
        loss.backward()
        opt.step()
        opt.zero_grad()

    return cs_loss, mmd_loss

def plot_results(data_ground_truth, data_new, data_new_initial, epochs, mmd_loss, cs_loss):
    #---------------------------------------
    # Ground truth vs initial representation
    #---------------------------------------
    plt.figure()
    plt.plot(data_ground_truth[:, 0],
             data_ground_truth[:, 1],
             'kx',
             label="Fixed representation")
    plt.plot(data_new_initial[:, 0],
             data_new_initial[:, 1],
             'b.',
             label="Optimized representation")
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.title("Before optimization")
    plt.legend(loc='upper left')
    plt.savefig(fname='representation_before_opt.pdf', bbox_inches='tight')
    #---------------------------------------
    # Ground truth vs final representation
    #---------------------------------------
    plt.figure()
    plt.plot(data_ground_truth[:, 0],
             data_ground_truth[:, 1],
             'kx',
             label="Fixed representation")
    
    plt.plot(data_new[:, 0],
             data_new[:, 1],
             'b.',
             label="Optimized representation")
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.title("After optimization with CS divergence")
    plt.legend(loc='upper left')
    plt.savefig(fname='representation_after_opt.pdf', bbox_inches='tight')
    #---------------------------------------
    # MMD loss
    #---------------------------------------
    plt.figure()
    plt.plot(range(epochs), mmd_loss, 'b')
    plt.xlabel('Iteration')
    plt.ylabel(r'$MMD^2$')
    #plt.savefig(fname='mmd.pdf', bbox_inches='tight')
    #---------------------------------------
    # CS loss
    #---------------------------------------
    plt.figure()
    plt.plot(range(epochs), cs_loss, 'k')
    plt.xlabel('Iteration')
    plt.ylabel(r'$D_{CS}$')
    #plt.savefig(fname='dcs.pdf', bbox_inches='tight')
    #---------------------------------------

    plt.show()

def main(cfg):
    data_new, data_ground_truth = init_data(cfg['n_data'],
                                            cfg['dim'],
                                            cfg['device'],
                                            cfg['gt_n_data'],
                                            cfg['gt_sigma'],
                                            cfg['gt_mean_1'],
                                            cfg['gt_mean_2'],)
    data_new_initial = data_new().detach().cpu().numpy().copy()

    cs_loss, mmd_loss = train(data_ground_truth,
                              data_new,
                              cfg['kernel_sigma'],
                              cfg['lr'],
                              cfg['epochs'])

    plot_results(data_ground_truth.cpu().numpy(),
                 data_new().detach().cpu().numpy(),
                 data_new_initial,
                 cfg['epochs'],
                 mmd_loss,
                 cs_loss)

if __name__ == '__main__':
    cfg = {
        'n_data': 400,
        'dim': 2,
        'gt_mean_1': [-4, -4],
        'gt_mean_2': [4, 4],
        'gt_sigma': 1,
        'gt_n_data': 200,
        'kernel_sigma': 1.,
        'lr': 200,
        'epochs': 10000,
        'device': get_device()
    }
    print(f"Using {cfg['device']} device")

    main(cfg)
    
    











