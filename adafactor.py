import torch
import torch.nn as nn
import operator
import functools
from copy import copy
from math import sqrt

class AdaFactor(torch.optim.Optimizer):
    def __init__(self, params, lr=None, beta1=0.9, beta2=0.999, eps1=1e-30, 
                 eps2=1e-3, cliping_threshold=1, relative_step_size=None,
                 non_constant_decay = True, enable_factorization=True,
                 weight_decay=0):
        enable_momentum =  False
        ams_grad = True
        self.beta1_glob = copy(beta1)
        self.beta2_glob = copy(beta2)
        self.lr_glob = copy(lr)
        
        if type(beta2) == 'function':
            enable_momentum = True
        elif beta1 != 0:
            enable_momentum = True
            if not 0.0 <= beta1 < 1.0:
                raise ValueError("Invalid beta parameter at index 0: {}".format(beta1))
            beta1 =lambda x: self.beta1_glob
        
        
        
        if non_constant_decay:
            ams_grad = False
            if not 0.0 <= beta2 < 1.0:
                raise ValueError("Invalid beta parameter at index 1: {}".format(beta2))
            beta1 = lambda t: self.beta1_glob * (1 - self.beta1_glob ** (t-1)) / (1 - self.beta1_glob ** t)
            beta2 = lambda t: self.beta2_glob * (1 - self.beta2_glob ** (t-1)) / (1 - self.beta2_glob ** t)
        elif type(beta2) != 'function':    
            beta2=lambda x: self.beta2_glob
        else:
            ams_grad = False    
                        
        
        relative_step_size  = False
        if lr is None:
            #default value from article
            lr = lambda t: min(1e-2, 1 / sqrt(t))
        elif type(lr) == 'function':
            relative_step_size  = True
            del self.lr_glob
        else:
            lr=lambda x: self.lr_glob
                         
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps1=eps1,
                         eps2=eps2, cliping_threshold=cliping_threshold,                                                           weight_decay=weight_decay,ams_grad=ams_grad,
                        enable_factorization=enable_factorization,
                        relative_step_size=relative_step_size,
                        enable_momentum=enable_momentum)
        
        super(AdaFactor, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdaFactor, self).__setstate__(state)       
     
    def _experimental_reshape(self,shape):
        temp_shape = shape[2:]
        if len(temp_shape) == 1:
            new_shape = (shape[0],shape[1]*shape[2])
        else:
            tmp_div = len(temp_shape) // 2 + len(temp_shape) % 2           
            new_shape = (shape[0]*functools.reduce(operator.mul, temp_shape[tmp_div:],1),
                         shape[1]*functools.reduce(operator.mul, temp_shape[:tmp_div],1))
        return new_shape, copy(shape)
        
        
    def _check_shape(self, shape):
        '''
        output1 - True - algorithm for matrix, False - vector;
        output2 - need reshape
        '''
        if len(shape) > 2:
            return True, True
        elif len(shape) == 2:
            return True, False
        elif len(shape) == 2 and (shape[0] == 1 or shape[1] == 1):
            return False, False
        else:
            return False, False
        
    def _rms(self, x):
        return sqrt(torch.mean(x.pow(2)))
    
    
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()       
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                cuda_id = -1
                if grad.is_cuda:
                    cuda_id = grad.get_device()
                    
                    
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead') 
                    
                flag1, flag2 = self._check_shape(grad.size())
                new_shape = p.data.size()
                if flag2 and group['enable_factorization']:
                    new_shape, old_shape =\
                    self._experimental_reshape(p.data.size())
                    grad = grad.view(new_shape)
               
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    if group['enable_momentum']:
                        state['exp_avg'] = torch.zeros(new_shape).cuda(cuda_id) if cuda_id >= 0 else  torch.zeros(new_shape)
                                           
                       
                    if flag1 and group['enable_factorization']:
                        state['exp_avg_sq_R'] = torch.zeros((1,new_shape[1])).cuda(cuda_id) if cuda_id >= 0 else torch.zeros((1,new_shape[1]))
                        state['exp_avg_sq_C'] = torch.zeros((new_shape[0],1)).cuda(cuda_id) if cuda_id >= 0 else torch.zeros((1,new_shape[1]))
                    else:
                        state['exp_avg_sq'] = torch.zeros(new_shape).cuda(cuda_id) if cuda_id >= 0 else torch.zeros(new_shape)
                    
                    if group['ams_grad']:
                        state['exp_avg_sq_hat'] = torch.zeros(new_shape).cuda(cuda_id) if cuda_id >= 0 else torch.zeros(new_shape)
                    
                
                if group['enable_momentum']:
                    exp_avg = state['exp_avg']
                    
                if flag1 and group['enable_factorization']:
                    exp_avg_sq_R = state['exp_avg_sq_R']
                    exp_avg_sq_C = state['exp_avg_sq_C'] 
                else:
                    exp_avg_sq = state['exp_avg_sq']
                
                if group['ams_grad']:
                    exp_avg_sq_hat = state['exp_avg_sq_hat']
                
                
                state['step'] += 1
                lr_t = group['lr'](state['step'])
                if group['relative_step_size']:
                    lr_t *= max(group['eps2'], self._rms(data))
                          
                if group['enable_momentum']:
                    beta1_t = group['beta1'](state['step'])
                    exp_avg.mul_(beta1_t).add_(1 - beta1_t, grad)
                    
                beta2_t = group['beta2'](state['step']) 

                if flag1 and group['enable_factorization']:
                    exp_avg_sq_R.mul_(beta2_t).add_(1 - beta2_t,                   
                      torch.sum(torch.mul(grad,grad).add_(group['eps1']), dim=0, keepdim=True))
                    exp_avg_sq_C.mul_(beta2_t).add_(1 - beta2_t,                   
                      torch.sum(torch.mul(grad,grad).add_(group['eps1']), dim=1, keepdim=True))
                    v = torch.mul(exp_avg_sq_C,exp_avg_sq_R).div_(torch.sum(exp_avg_sq_R))
                else:
                    exp_avg_sq.mul_(beta2_t).addcmul_(1 - beta2_t, grad, grad).add_((1 - beta2_t)*group['eps1'])
                    v = exp_avg_sq

                
                g = grad
                if group['enable_momentum']:
                    g = torch.div(exp_avg,1 - beta1_t ** state['step'])
                               
                if group['ams_grad']:
                    torch.max(exp_avg_sq_hat, v, out=exp_avg_sq_hat)
                    v = exp_avg_sq_hat                    
                    u = torch.div(g,(torch.div(v,1 - beta2_t ** state['step'])).sqrt().add_(group['eps1']))
                else:
                    u = torch.div(g,v.sqrt()) 
                    
                u.div_(max(1,self._rms(u) / group['cliping_threshold']))
                p.data.add_(-lr_t * (u.view(old_shape) if flag2 and group['enable_factorization'] else u))
            
 
                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * lr_t, p.data)
                    
        return loss
