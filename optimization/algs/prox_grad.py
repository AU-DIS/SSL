from numpy import iterable
from torch import Tensor
from torch.optim.sgd import SGD

#class that inherets from the SGD = Stochastic Gradient Descent
class PGM(SGD):
    def __init__(self,
                 params,
                 proxs,
                 lr: float = 0.2,
                 momentum: float = 0,
                 dampening: float = 0,
                 nesterov: bool = False):
        #NOT USED FOR THE EXPERIMENTS
        #--------------------------------------------------
        if momentum != 0:
            raise ValueError("momentum is not supported")
        if dampening != 0:
            raise ValueError("dampening is not supported")
        if nesterov != 0:
            raise ValueError("nesterov is not supported")
        #--------------------------------------------------

        kwargs = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=0,
                      nesterov=nesterov)
        super().__init__(params, **kwargs)

        if len(proxs) != len(self.param_groups):
            raise ValueError(
                "Invalid length of argument proxs: {} instead of {}".format(len(proxs),
                                                                            len(self.param_groups)))

        for group, prox in zip(self.param_groups, list(proxs)):
            group.setdefault('prox', prox) #The setdefault() method returns the value of the item with the specified key.

    #Overwrite the step function that -> Performs a single optimization step
    def step(self, lamb, closure=None):#lamb = 0 AND closure is None for our experiments
        # this performs a gradient step
        # optionally with momentum or nesterov acceleration
        self.param_groups[0]['params']
        super().step(closure=closure) #here it calls the step of the SGD

        for group in self.param_groups:
            prox = group['prox'] #v_prox': ProxNonNeg() 

            # here we apply the proximal operator to each parameter in a group
            for p in group['params']:
                p.data = prox(z=p.data, lamb=lamb)
