import torch
from torch.autograd import Variable

class SuperRelu(torch.autograd.Function):
    def forward(self, input_):
        self.save_for_backward(input_)
        output = input_.clamp(min=0, max=1)
        return output

    def backward(self, grad_outputs):
        input_ = self.saved_tensors
        grad_input = grad_outputs.clone()
        grad_input[input_ < 0] = 0
        grad_input[input_ > 1] = 0
        return grad_input

input_ = Variable(torch.randn(3, 3))
srelu = SuperRelu()
output = srelu(input_)
print(output.data)