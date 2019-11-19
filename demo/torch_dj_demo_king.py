import torch
from torch.autograd import Variable

class SuperRelu(torch.autograd.Function):
    def forward(self, inp, x_dim):
        self.save_for_backward(inp)
        dim = x_dim.item()
        output = input_.clone()
        output[:, :dim] = output[:, :dim].clamp(min=0, max=1)
        return output

    def backward(self, grad_outputs):
        inp = self.saved_tensors
        inp = inp[0]
        grad_input = grad_outputs.clone()
        grad_input[inp < 0] = 0
        grad_input[inp > 1] = 0
        return grad_input, None

input_ = Variable(torch.randn(3, 3, requires_grad=True))
w = Variable(torch.randn(3, 1),requires_grad=True)
srelu = SuperRelu()
output = srelu(input_ * w, torch.IntTensor([2])).sum()
print(input_.data)
print(output.data)
output.backward()
