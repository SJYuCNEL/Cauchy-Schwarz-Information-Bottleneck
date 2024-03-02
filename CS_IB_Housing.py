import torch
import torch.nn
import torch.nn.init
import torchvision
import argparse
import PIL.Image
import sklearn.datasets
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import sys


from loss_cs import CS_Div, CS_QMI, CS_QMI_normalized
from kde_estimation_mi import KDE_IXT_estimation
#torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import math
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description='PyTorch Phase amplitude retrieval ')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')


parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate')
                    
parser.add_argument('--beta', type=float, default=0.01, metavar='beta',
                    help='beta')


parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
                   
args = parser.parse_args()

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class DatasetRegression(torch.utils.data.Dataset):

    def __init__(self,X,Y):
        self.data = X
        self.targets = Y
    
    def __getitem__(self,index):
        data, target = self.data[index], self.targets[index]
        return data, target

    def __len__(self):
        return len(self.targets)
        
        
def get_california_housing(percentage_validation=0.2,percentage_test=0.2):

    X, Y = sklearn.datasets.fetch_california_housing(data_home='../datas/CaliforniaHouring/', \
        download_if_missing=True,return_X_y=True)
    
    # We remove the houses with prices higher than 500,000 dollars
    idx_drop = Y >= 5
    X, Y = X[~idx_drop], np.log(Y[~idx_drop])
    #X, Y = X[~idx_drop], Y[~idx_drop]

    # We shuffle the inputs and outputs before assigning train/val/test 
    tmp = list(zip(X,Y))
    random.shuffle(tmp)
    X, Y = zip(*tmp)
    X, Y = torch.FloatTensor(X), torch.FloatTensor(Y)
    X = (X - torch.mean(X,0)) / torch.std(X,0)
    #Y = (Y - torch.mean(Y,0)) / torch.std(Y,0)

    # Split between training / validation / testing
    splitpoint_test = int(len(Y) * (1.0 - percentage_test))
    splitpoint_validation = int(splitpoint_test * (1.0 - percentage_validation))
    X_train, Y_train = X[:splitpoint_validation], Y[:splitpoint_validation]
    X_validation, Y_validation = X[splitpoint_validation:splitpoint_test], Y[splitpoint_validation:splitpoint_test]
    X_test, Y_test = X[splitpoint_test:], Y[splitpoint_test:]

    # Generate and return the datasets
    trainset = DatasetRegression(X_train,Y_train)
    validationset = DatasetRegression(X_validation,Y_validation)
    testset = DatasetRegression(X_test,Y_test)
    return trainset,validationset, testset
    
    
    
class Deterministic_encoder(torch.nn.Module):
    '''
    Probabilistic encoder of the network.
    - We use the one in Kolchinsky et al. 2019 "Nonlinear Information Bottleneck"
    - Parameters:
        · K (int) : dimensionality of the bottleneck variable
        · n_x (int) : dimensionality of the input variable
    '''

    def __init__(self,K,n_x):
        super(Deterministic_encoder,self).__init__()

        self.K = K
        self.n_x = n_x

        layers = []
        layers.append(torch.nn.Linear(n_x,128))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(128,128))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(128,self.K))
        self.f_theta = torch.nn.Sequential(*layers)

    def forward(self,x):

        x = x.view(-1,self.n_x)
        mean_t = self.f_theta(x)
        return mean_t

class Deterministic_decoder(torch.nn.Module):
    '''
    Deterministic decoder of the network.
    - We use the one in Kolchinsky et al. 2019 "Nonlinear Information Bottleneck"
    - Parameters:
        · K (int) : dimensionality of the bottleneck variable
        · n_y (int) : dimensionality of the output variable (number of classes)
    '''

    def __init__(self,K,n_y):
        super(Deterministic_decoder,self).__init__()

        self.K = K

        layers = []
        layers.append(torch.nn.Linear(self.K,128))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(128,n_y))
        self.g_theta = torch.nn.Sequential(*layers)

    def forward(self,t):

        logits_y = self.g_theta(t).squeeze()
        return logits_y
    

class CS_IB_network(torch.nn.Module):
    '''
    - We use the one in Kolchinsky et al. 2019 "Nonlinear Information Bottleneck"
    - Parameters:
        · K (int) : dimensionality of the bottleneck variable
        · n_x (int) : dimensionality of the input variable
        · n_y (int) : dimensionality of the output variable (number of classes)
        · train_logvar_t (bool) : if true, logvar_t is trained
    '''

    def __init__(self,K,n_x,n_y,logvar_t=-1.0,train_logvar_t=False):
        super(CS_IB_network_,self).__init__()

        self.encoder = Deterministic_encoder(K,n_x)
        self.decoder = Deterministic_decoder(K,n_y)
        self.enc_mean = torch.nn.Linear(K, K)
        self.enc_std = torch.nn.Linear(K, K)


    def encode(self,x,training=True):

        t = self.encoder(x)
        enc_mean, enc_logvar = self.enc_mean(t), self.enc_std(t)
        std = torch.exp(enc_logvar / 2)
        eps = torch.randn_like(std)

        
        if training:
            latent = torch.add(torch.mul(std, eps), enc_mean)
        else:
            latent = enc_mean
        return latent
    
    def apply_noise(self,mean_t):
        return mean_t + torch.exp(0.5*self.logvar_t) * torch.randn_like(mean_t)

    def decode(self,t):

        logits_y = self.decoder(t)
        return logits_y

    def forward(self,x,training):
        t = self.encode(x, training)
        logits_y = self.decode(t)
        return t, logits_y
    

        
def train(model, trainloader,criterion,optimizer,epoch, beta, varY):
    #global min_loss
    ITY_train = 0.0
    ITX_train = 0.0
    cs_div_train = 0.0
    total_loss = 0.0
    
    model.train()

    for train_x, train_y in trainloader:
        
        train_x = train_x.to(device)
        train_y = train_y.to(device) 
        
        z, output = model(train_x,training=False) #Forward pass
       
        #Compute losses
        
            

        mse = criterion(output,train_y)
        test_IZY = 0.5 * torch.log(varY / mse) / np.log(2) # in bits
        
        IXZ = CS_QMI_normalized(train_x.view(train_x.shape[0],-1), z, sigma=1)

        
        CS_div = CS_Div(train_x.view(train_x.shape[0],-1),train_y,output,sigma=1)
        loss = CS_div +  beta*IXZ
            

        
        #Zero current grads and do backprop
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        
      

        ITY_train += test_IZY.item()
        ITX_train += IXZ.item()
        
        cs_div_train += CS_div.item()
        total_loss += loss.item()
   
   

    print('Train Epoch: {} IXZ_train: {:.4f} IYZ_train: {:.4f} cs_div_train: {:.4f} total_train: {:.4f}'.format(
                    epoch, ITX_train/len(trainloader),ITY_train/len(trainloader),
                    cs_div_train/len(trainloader),total_loss/len(trainloader))) 
   
    return ITX_train/len(trainloader), ITY_train/len(trainloader)

    
    
def test(model, testloader,criterion,epoch, beta, varY):
    #global min_loss
    ITY_test = 0.0
    ITX_test = 0.0
    cs_div_test = 0.0
    total_loss = 0.0
    
    model.eval()
    with torch.no_grad():
        for test_x, test_y in testloader:

            test_x = test_x.to(device)
            test_y = test_y.to(device) 

            z, output = model(test_x, training=False) #Forward pass

            #Compute losses
          

            mse = criterion(output,test_y)
            test_IZY = 0.5 * torch.log(varY / mse) / np.log(2)
            
            IXZ = CS_QMI_normalized(test_x.view(test_x.shape[0],-1), z, sigma=1)
        
            CS_div = CS_Div(test_x.view(test_x.shape[0],-1),test_y,output,sigma=1)
            loss = mse +  beta*IXZ

         

            ITY_test += test_IZY.item()
            ITX_test += IXZ.item()
            cs_div_test += CS_div.item()
            total_loss += loss.item()

        print('Test Epoch: {} IXZ_test: {:.4f} IYZ_test: {:.4f} CS_div_test: {:.4f} total_test: {:.4f}'.format(
                    epoch, ITX_test/len(testloader),ITY_test/len(testloader), 
                    cs_div_test/len(testloader),total_loss/len(testloader)))

        
    return ITX_test/len(testloader), ITY_test/len(testloader)

def test_rmse(model,test_loader):
    model.eval()
    with torch.no_grad():
        for test_x, test_y in test_loader:
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            _,output_pre = model(test_x,training=False)
    output_numpy = output_pre.cpu().numpy()
    target_numpy = test_y.cpu().numpy()
    rmse =  np.sqrt(np.mean((output_numpy-target_numpy)**2))
    print('test_RMSE', rmse)
    return rmse
    
trainset,validationset, testset = get_california_housing(percentage_validation=0.2,percentage_test=0.2)



betas = [0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1,5e-1, 1]

for beta in betas:
    save_path = './xxx_'+str(beta)
    PATH = save_path +'/cs_IB_QMI_norm.pth'
    if not os.path.isdir(save_path):
      os.makedirs(save_path,exist_ok=True)
      
    best_test_rmse = 1
    IXT_train_all = []
    IYT_train_all = []
    rmse_test_all = []
    print("###################################################")
    print("beta", beta)
    print("###################################################")
    
    sgd_batch_size = 128
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=sgd_batch_size,shuffle=True,num_workers=8)
    validation_loader = torch.utils.data.DataLoader(validationset,batch_size=sgd_batch_size,shuffle=False,num_workers=8)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset),shuffle=False,num_workers=8)
    varY = torch.var(trainset.targets)

    n_x = 8
    n_y = 1
    K = 128
    network =  CS_IB_network_(K,n_x,n_y).to(device)

    # Definition of the optimizer
    learning_rate=0.0001
    optimizer = torch.optim.Adam(network.parameters(),lr=learning_rate)

    criterion = torch.nn.MSELoss()
    learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, \
                step_size=10,gamma=0.6)
    n_epochs = 200
    
    for epoch in range(n_epochs):
        IXT_train, ITY_train = train(network, train_loader,criterion,optimizer,epoch, beta, varY)
        IXT_test, ITY_test = test(network, validation_loader,criterion,epoch, beta, varY)
        #get_MI_QMI(train_loader, network)
        rmse = test_rmse(network,test_loader)
        
        if rmse<best_test_rmse:
          print('saving')
          state = {
            'net': network.state_dict(),
            'IXT_train': IXT_train,
            'ITY_train': ITY_train,
            'best_rmse': rmse
                  }
          torch.save(state, PATH)
          best_test_rmse = rmse
          
        rmse_test_all.append(rmse)
        IXT_train_all.append(IXT_train)
        IYT_train_all.append(ITY_train)
    
    best_rmse = np.array(rmse_test_all).min()
      
    
    PATH_IXT = save_path +'/IXT'
    PATH_IYT = save_path +'/IYT'
    #torch.save(state, PATH)
    
    IXT_train_numpy = np.array(IXT_train_all)
    IYT_train_numpy = np.array(IYT_train_all)
    np.save(PATH_IXT, IXT_train_numpy)
    np.save(PATH_IYT, IYT_train_numpy)
    print('best RMSE', np.array(rmse_test_all).min())
    print("###################################################")

