# adafactor-pytorch
A pytorch realization of adafactor  (https://arxiv.org/pdf/1804.04235.pdf )

# Notes
1)Factorization works on any dimension. When dimension of weight tensor is higher than 2, it will be reshaped to 2D. For turning  off this feature  just change this lines ( if len(shape) > 2: return False, True ) in _check_shape 

2)Weights decay was moved to proper position according (https://arxiv.org/abs/1711.05101 )

# Parameters description:
lr - learning rate can be scalar or function, in second case relative step size is using.

beta1, beta2 - is also can be scalar or functions, in first case algorithm works as AMSGrad. Setting beta1 to zero is turning off moments updates.

non_constant_decay - boolean, has effect if betas are scalars. If True using functions for betas (from section 7.1)

enable_factorization - boolean. Factorization works on 2D weights.

clipping_threshold - scalar. Threshold value for update clipping (from section 6)
