import torch
import numpy as np
from numpy import ndarray
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def Torch_global_set():
    print(torch.cuda.is_available())
    torch.manual_seed(42)

def gen_2d_tensor(x:int, y:int) -> Tensor:
    a = torch.randn(x, y)
    return a

def gen_3d_tensor(x:int, y:int, z:int) -> Tensor:
    a = torch.randn(x, y, z)
    return a

def tensor2ndarray(a:Tensor) -> ndarray:
    a_np = np.array(a)
    print("1:", a_np[0, 0])
    return a_np

def ndarray2tensor(a:ndarray) -> Tensor:
    a_tensor = torch.from_numpy(a)
    print("2:", a_tensor[0, 0])
    return a_tensor

def get_tensor_stats(a:Tensor):
    print("size of a:", a.size())
    print("mean value of a:", a.mean())
    print("std of a:", a.std())
    print("median value of a:", a.median())

def get_tensor_mean_by_axis(a:Tensor):
    b2 = torch.mean(input=a, dim=2, keepdim=True)
    b12 = torch.mean(input=a, dim=(1, 2), keepdim=False)
    print("size of b2:", b2.size())
    print("size of b12:", b12.size())

def flatten_tensor_by_axis(a, model_tmp:nn.Module):
    a_f = torch.flatten(a, start_dim=-2, end_dim=-1)
    output_1_6 = model_tmp(a_f)
    print(output_1_6)

def calc_mse_loss(output):
    all_1 = torch.ones_like(output)
    mse = F.mse_loss(output, all_1)
    print("MSE between two tensors:", mse)
    return mse

def back_propagation(model_tmp:nn.Module, output):
    mse = calc_mse_loss(output)
    # clear gradients
    model_tmp.zero_grad()
    # back-propagation
    mse.backward()
    # extract weights of the last layer
    w_last_layer = list(model_tmp.parameters())[-1]
    # extract gradients of the last layer
    grd = w_last_layer.grad
    print(grd[:, 5])
    return grd

def create_sgd(model_tmp:nn.Module, grd:Tensor):
    optimizer = optim.SGD(model_tmp.parameters(), lr=1e-3)
    # step the optimizer
    optimizer.zero_grad()
    optimizer.step()
    # compute the variation of params of the last layer
    #vari = torch.abs(list(model_tmp.parameters())[-1] - (1e-3 * grd))
    vari = torch.abs(1e-3 * grd / list(model_tmp.parameters())[-1])
    vari = torch.round(vari, decimals=3)
    torch.set_printoptions(sci_mode=False)
    print(vari)