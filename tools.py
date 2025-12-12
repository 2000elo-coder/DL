import torch
from torch.autograd import Variable

def from_numpy_to_tensor(numpys, require_grads, dtype=torch.float32):
    outputs = list()
    for ind in range(len(numpys)):
        outputs.append(
            Variable(torch.from_numpy(numpys[ind]), requires_grad=require_grads[ind]).type(dtype)
        )
    return outputs