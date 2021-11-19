import torch

opt_param_example = torch.rand(200).requires_grad_()
#set requires grad for parameters that have to be optimized

other_param =  torch.rand(200)


#use newly added function here
def example_function_to_be_tested(opt_param_, other_param_):
    return  opt_param_+other_param_


#create scalar loss function by summming
loss = torch.sum(example_function_to_be_tested(opt_param_example, other_param))

#try to calculate derivative with respect to parameter and check that it is non_zero
print(torch.sum(torch.abs(torch.autograd.grad(loss, opt_param_example)[0]))>0)
