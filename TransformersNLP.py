import torch
import transformers
import wandb

x = torch.rand(3)

print(x)


class ClassName(object):
    '''docstring for ClassName.'''

    def __init__(self, arg):
        super(ClassName, self).__init__()
        self.arg = arg

    def mname(self, arg):
        pass
