import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
import byzantine1

def fltrust(gradients, net, lr, f, byz, local_weighted_score):
    """
    gradients: list of gradients. The last one is the server update.
    net: model parameters.
    lr: learning rate.
    f: number of malicious clients. The first f clients are malicious.
    byz: attack type.
    """
    
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    # let the malicious clients (first f clients) perform the byzantine attack
    param_list = byz(param_list, net, lr, f)
    n = len(param_list) - 1
    
    # use the last gradient (server update) as the trusted source
    baseline = nd.array(param_list[-1]).squeeze()
    cos_sim = []
    new_param_list = []
    
    # compute cos similarity
    for each_param_list in param_list:
        each_param_array = nd.array(each_param_list).squeeze()
        cos_sim.append(nd.dot(baseline, each_param_array) / (nd.norm(baseline) + 1e-9) / (nd.norm(each_param_array) + 1e-9))

        
    cos_sim = nd.stack(*cos_sim)[:-1]
    cos_sim = nd.maximum(cos_sim, 0) # relu
    normalized_weights = cos_sim / (nd.sum(cos_sim) + 1e-9) # weighted trust score

    # normalize the magnitudes and weight by the trust score
    for i in range(n):
        new_param_list.append(param_list[i] * normalized_weights[i] / (nd.norm(param_list[i]) + 1e-9) * nd.norm(baseline))
    
    # update the global model
    global_update = nd.sum(nd.concat(*new_param_list, dim=1), axis=-1)
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        param.set_data(param.data() - lr * global_update[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size       

def fedavgfl(gradients, net, lr, f, byz, local_weighted_score):
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    # let the malicious clients (first f clients) perform the byzantine attack
    param_list = byz(param_list, net, lr, f)
    n = len(param_list)
    new_param_list = []
    # normalize avg
    for i in range(n):
        new_param_list.append(param_list[i] * local_weighted_score[i])
    # update the global model
    global_update = nd.sum(nd.concat(*new_param_list, dim=1), axis=-1)
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        param.set_data(param.data() - lr * global_update[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size  
#########WaterMark#############
#single worker local training
def single_localupdate(gradients, net, lr, f, byz, i):
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    if i < f and byz == byzantine1.trim_attack: # see if it is a malicious client
        print('param_list before', param_list[0][0:1])
        #param_list = byzantine1.trim_attack_local(param_list, net, lr)[0]
        param_list = byz(param_list, net, lr, 1)
        print('param_list after', param_list[0][0:1])
    local_update = nd.sum(nd.concat(*param_list, dim=1), axis=-1)
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        param.set_data(param.data() - 1/64 * lr * local_update[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size 

def attack_param(gradients, net, lr, f, byz):
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    # let the malicious clients (first f clients) perform the byzantine attack
    #print('param_list before', param_list[0][0])
    param_list = byz(param_list, net, lr, f)
    #print('param_list after', param_list[0][0])
    return param_list

def extract_locals(param_list_after,net, lr, index):
    spe_param_list = []
    spe_param_list = param_list_after[index]
    #print('spe_param_list',spe_param_list)
    # return the specific local model
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        param.set_data(param.data() - 0.0001 * spe_param_list[idx:(idx+param.data().size)].reshape(param.data().shape)) #0.00002
        idx += param.data().size  

def watermarkfl(param_list_after, net, lr, local_weighted_score, trust_value):
    #param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    #aggregate local models
    n = len(param_list_after)
    #print('n,sum_local_weighted_score',n,sum(local_weighted_score))
    new_param_list = []
    # normalize avg
    for i in range(n):
        new_param_list.append(param_list_after[i] * local_weighted_score[i] * trust_value[i])
    #print('new_param_list[0]',new_param_list[0])
    #print('new_param_list[9]',new_param_list[9])
    # update the global model
    global_update = nd.sum(nd.concat(*new_param_list, dim=1), axis=-1)
    #print('global_update',global_update)
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        param.set_data(param.data() - lr * global_update[idx:(idx+param.data().size)].reshape(param.data().shape))
        idx += param.data().size  
