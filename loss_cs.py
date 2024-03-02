import numpy as np
import torch
import torch.nn as nn

def sigma_estimation(X, Y):
    """ sigma from median distance
    """
    D = distmat(torch.cat([X,Y]))
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med=np.mean(Tri)
    if med<1E-2:
        med=1E-2
    return med

def distmat(X):
    """ distance matrix
    """

    r = torch.sum(X*X, 1)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X,0,1))
    D = r.expand_as(a) - 2*a +  torch.transpose(r,0,1).expand_as(a)
    D = torch.abs(D) 
    return D

def GaussianMatrix(X,Y,sigma):
    size1 = X.size()
    size2 = Y.size()
    G = (X*X).sum(-1)
    H = (Y*Y).sum(-1)
    Q = G.unsqueeze(-1).repeat(1,size2[0])
    R = H.unsqueeze(-1).T.repeat(size1[0],1)
    
    
    H = Q + R - 2*X@(Y.T)
    H = torch.clamp(torch.exp(-H/2/sigma**2),min=0)
    
    
    return H


def CS_Div(x,y1,y2,sigma): # conditional cs divergence Eq.18
    K = GaussianMatrix(x,x,sigma)
    L1 = GaussianMatrix(y1,y1,sigma)
    L2 = GaussianMatrix(y2,y2,sigma)
    L21 = GaussianMatrix(y2,y1,sigma);

    H1 = K*L1
    self_term1 = (H1.sum(-1)/(K**2).sum(-1)).sum(0)
    
    H2 = K*L2
    self_term2 = (H2.sum(-1)/(K**2).sum(-1)).sum(0)
    
    H3 = K*L21;
    cross_term = (H3.sum(-1)/(K**2).sum(-1)).sum(0)
    
    return -2*torch.log2(cross_term) + torch.log2(self_term1) + torch.log2(self_term2)


def CS_QMI(x,y,sigma = None):
    """
    x: NxD
    y: NxD
    Kx: NxN
    ky: NxN
    """
    
    N = x.shape[0]
    #print(N)
    if not sigma:
        sigma_x = 10*sigma_estimation(x,x)
        sigma_y = 10*sigma_estimation(y,y)
       
        Kx = GaussianMatrix(x,x,sigma_x)
        Ky = GaussianMatrix(y,y,sigma_y)
    
    else:
        Kx = GaussianMatrix(x,x,sigma)
        Ky = GaussianMatrix(y,y,sigma)
    
    #first term
    self_term1 = torch.trace(Kx@Ky.T)/(N**2)
    
    #second term  
    self_term2 = (torch.sum(Kx)*torch.sum(Ky))/(N**4)
    
    #third term
    term_a = torch.ones(1,N).to(x.device)
    term_b = torch.ones(N,1).to(x.device)
    cross_term = (term_a@Kx.T@Ky@term_b)/(N**3)
    CS_QMI = -2*torch.log2(cross_term) + torch.log2(self_term1) + torch.log2(self_term2)
    
    return CS_QMI

def KDE_KL(y,y_pre,sigma=1):
    G_y_y = GaussianMatrix(y.view(-1,1),y.view(-1,1),sigma)
    G_y_y_pre = GaussianMatrix(y.view(-1,1),y_pre.view(-1,1),sigma)
    
    log_G = torch.log(torch.sum(G_y_y,dim=0))-torch.log(torch.sum(G_y_y_pre,dim=0))
    return torch.mean(log_G)




def CS_QMI_normalized(x,y,sigma):

    QMI = CS_QMI(x, y, sigma)
    var1 = torch.sqrt(CS_QMI(x, x, sigma))
    var2 = torch.sqrt(CS_QMI(y, y, sigma))
    
    return QMI/(var1*var2)
    
    
def FGSM(inputs, target, device, model, eps, types):
    #model.eval()
    model.train() 
    inputs = inputs.clone().detach().to(device)
    target = target.clone().detach().to(device)
    loss = nn.MSELoss()
    inputs.requires_grad = True
    if types=='CS_IB':
        _, outputs = model(inputs, training=False)
    elif types=='NIB':
        outputs = model(inputs)
    elif types =='HSIC':
        _,outputs = model(inputs)
    if not outputs.shape:
        cost = loss(outputs.view(1), target)
    else:
        cost = loss(outputs, target)
    # # Update adversarial images
    grad = torch.autograd.grad(cost, inputs,
                               retain_graph=False, create_graph=False)[0]

    adv_images = inputs + eps*grad.sign()
    #adv_images = torch.clamp(adv_images, min=0, max=1).detach()
    return adv_images.detach()


def PGD(inputs, target, model, eps=0.3, alpha=0.05, iters=5, types='CS_IB') :
    model.eval()
    inputs = inputs.clone().detach().to(device)
    target = target.clone().detach().to(device)
    loss = nn.MSELoss()
    
    adv_images = inputs.clone().detach()
        
    for i in range(iters) :    
        adv_images.requires_grad = True
        if types == 'CS_IB':
            _, outputs = model(adv_images[None,:], training=False)
        elif types=='NIB':
            outputs = model(adv_images[None,:])
        elif types=='HSIC':
            _, outputs = model(adv_images[None,:])

        if not outputs.shape:
            cost = loss(outputs.view(1), target)
        else:
            cost = loss(outputs, target)
        
        grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - inputs,
                            min=-eps, max=eps)
        adv_images = inputs + delta
            
    return adv_images.detach()   
    
    
def FGSM_attacks(network,testset,device,types):
    for eps_value in [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6]:
        

        test_loader = torch.utils.data.DataLoader(testset,batch_size=1,shuffle=False)
        output_numpy_list = []
        target_numpy_list = []
        network.eval()

        for test_x, test_y in test_loader:
            adv_example = FGSM(test_x, test_y, device, network, eps=eps_value, types=types)

            with torch.no_grad():
                if types=='CS_IB':
                    _,output_pre = network(adv_example.to(device), training=False)
                elif types=='NIB':
                    output_pre = network(adv_example.to(device))
                elif types=='HSIC':
                    _,output_pre = network(adv_example.to(device))
                


            output_numpy = output_pre.cpu().numpy()
            output_numpy_list.append(output_numpy)
            target_numpy = test_y.cpu().numpy()
            target_numpy_list.append(target_numpy)

        predict_numpy = np.array(output_numpy_list).reshape(-1,1)
        target_gd = np.array(target_numpy_list)
        rmse =  np.sqrt(np.mean((predict_numpy-target_gd)**2))

        print('eps', eps_value)
        print('RMSE', rmse) 
