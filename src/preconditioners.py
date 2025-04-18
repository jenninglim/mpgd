import torch
from abc import abstractmethod

DEBUG = False


class Preconditioner:
    def __init__(self, model, alpha=0.9):
        self.model = model
        self.g2 = {}
        self.alpha = alpha
        for name, param in model.named_parameters():
            self.g2[name] = torch.ones_like(param)

    def step(self,):
        self._step()

    @abstractmethod
    def _step(self,):
        '''
        Update gradient and preconditioner
        '''
        raise NotImplementedError()


class RMSPropPreconditioner(Preconditioner):
    def __init__(self, model, alpha=0.99):
        super().__init__(model, alpha)
    
    def _step(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not param.grad.isnan().any(), "NaN in gradient"
                self.g2[name] = self.alpha * self.g2[name] + (1 - self.alpha) * param.grad ** 2
                assert not self.g2[name].isnan().any()
                param.grad.data /= (self.g2[name] + 1e-8).sqrt()
                assert not param.grad.isnan().any()


class ConstantPreconditioner(Preconditioner):
    def __init__(self, model, c=1.):
        super().__init__(model, 0)
        self.c = c
    
    def _step(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad.data /= self.c

class AdamPreconditioner(Preconditioner):
    '''
    Adam-like preconditioner
    '''
    def __init__(self, model, alpha=0.9):
        super().__init__(model, alpha)
        self.t = 0
    
    def _step(self):
        self.t = self.t + 1
        t = self.t
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not param.grad.isnan().any(), "NaN in gradient"
                temp = self.alpha * self.g2[name] + (1 - self.alpha) * param.grad ** 2
                if DEBUG:
                    print("{0} grad norm {1}".format(name, param.grad.norm().item()))
                param.grad.data /= (self.g2[name].sqrt() + 1e-8) / (1 - self.alpha ** t)
                if DEBUG:
                    print("{0} grad norm {1}".format(name, param.grad.norm().item()))
                assert not temp.isnan().any(), f"""
                    NaN in preconditioner {temp.isnan().any()}, {self.g2[name].isnan().any()}, {param.grad.isnan().any()}
                    """
                assert not param.grad.isnan().any(), "NaN found in updated"
                self.g2[name] = temp
                if DEBUG:
                    print(self.g2[name].max(), self.g2[name].min())

PRECONDITIONERS = {'adam': AdamPreconditioner,
                   'rmsprop': RMSPropPreconditioner,
                   'constant': ConstantPreconditioner}