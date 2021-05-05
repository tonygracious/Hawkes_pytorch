import torch
from Utils.loss import MaxLogLike, fill_triu
import torch.nn.functional as F

class PointProcessModel(torch.nn.Module):
    """
    The class of generalized Hawkes process model
    contains most of necessary function.
    """
    def __init__(self, num_type: int):
        """
        Initialize generalized Hawkes process
        :param num_type: int, the number of event types.
        """
        super().__init__()
        self.model_name = 'A Poisson Process'
        self.num_type = num_type
        self.mu = torch.nn.Parameter(torch.randn(self.num_type) * 0.5 - 2.0)
        self.loss_function = MaxLogLike()


class MultiVariateHawkesProcessModel(PointProcessModel):
    """
        The class of generalized Hawkes process model
        contains most of necessary function.
        """

    def __init__(self, num_type: int, num_decay : int):
        """
        Initialize generalized Hawkes process
        :param num_type: int, the number of event types.
        :param num_decay: int, the number of decay functions
        """
        super().__init__(num_type)
        self.model_name = 'A MultiVariate Hawkes Process'
        self.num_decay = num_decay
        self.alpha = torch.nn.Parameter(torch.randn(self.num_type, self.num_type, self.num_decay) * 0.5 - 3.0)
        self.beta =  torch.nn.Parameter(torch.randn(self.num_decay) * 0.5 - 3.0)

    def forward(self, event_times, event_types, input_mask, t0, t1):
        """
        :param event_times:  B x N
        :param input_mask:  B x  N
        :param t0: starting time
        :param t1: ending time
        :return: loglikelihood
        """

        mhat = F.softplus(self.mu)
        Ahat = F.softplus(self.alpha)
        omega  = F.softplus(self.beta)

        B, N = event_times.shape
        dim = mhat.shape[0]

        # compute m_{u_i}
        mu = mhat[event_types]  # B x N


        # diffs[i,j] = t_i - t_j for j < i (o.w. zero)
        dt = event_times[:, :, None] - event_times[:, None]  # (N, T, T)
        dt = fill_triu(dt, 0)
        dt = dt.unsqueeze(1)  # (N,1,T,T)

        # kern[i,j] = omega* torch.exp(-omega*dt[i,j])
        kern = omega.view(1,-1, 1, 1) * torch.exp(-omega.view(1,-1, 1, 1) * dt) #(N, num_decay, T, T)

        colidx = event_types.unsqueeze(1).repeat(1, N, 1) #(N, T, T)
        rowidx = event_types.unsqueeze(2).repeat(1, 1, N) #(N, T, T)

        Auu = Ahat[rowidx, colidx].permute(0, 3, 1, 2) #(N, num_decay, T, T )

        ag = Auu * kern #(N, num_decay, T, T )
        ag = torch.tril(ag, diagonal=-1) # (N, num_decay, T, T ) lower triangular entries of (T, T) matrices

        # compute total rates of u_i at time i
        rates = mu + torch.sum(torch.sum(ag, dim=3), dim=1) #(N, T )

        #baseline \sum_i^dim \int_0^T \mu_i
        compensator_baseline = (t1 - t0) * torch.sum(mhat) #(1)

        # \int_{t_i}^T \omega \exp{ -\omega (t - t_i )  }
        log_kernel = -omega.view(1, -1, 1) * (t1[:, None] - event_times).unsqueeze(dim=1)  # (N, 2, T )
        Int_kernel = (1 - torch.exp(log_kernel))

        Au = Ahat[:, event_types].permute(1, 0, 3, 2) #(N, num_decay, num_decay, T )

        Au_Int_kernel = (Au * Int_kernel.unsqueeze(dim=1)).sum(dim=1).sum(dim=1)  * input_mask #(N, T)

        compensator = compensator_baseline + Au_Int_kernel.sum(dim=1) #(N, 1)

        loglik = torch.log(rates + 1e-8).mul(input_mask).sum(-1)  # (N, 1)

        return (loglik, compensator) # ((N,1), (N,1))



class HawkesPointProcess(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.mu = torch.nn.Parameter(torch.zeros(1)) #torch.nn.Parameter(torch.randn(1) * 0.5 - 2.0)
        self.alpha = torch.nn.Parameter(torch.zeros(1)) #torch.nn.Parameter(torch.randn(1) * 0.5 - 3.0)
        self.beta = torch.nn.Parameter(torch.zeros(1)) #torch.nn.Parameter(torch.randn(1) * 0.5)

    def logprob(self, event_times, input_mask, t0, t1):
        """
        :param event_times:
        :param input_mask:
        :param t0:
        :param t1:
        :return:
        """

        mu = F.softplus(self.mu)
        alpha = F.softplus(self.alpha)
        beta = F.softplus(self.beta)

        dt = event_times[:, :, None] - event_times[:, None]  # (N, T, T)
        dt = fill_triu(-dt * beta, -1e20)
        lamb = torch.exp(torch.logsumexp(dt, dim=-1)) * alpha * beta + mu  # (N, T)
        loglik = torch.log(lamb + 1e-8).mul(input_mask).sum(-1)  # (N,)

        log_kernel = -beta * (t1[:, None] - event_times) * input_mask + (1.0 - input_mask) * -1e20

        compensator = (t1 - t0) * mu
        compensator = compensator - alpha  * (torch.exp(torch.logsumexp(log_kernel, dim=-1)) - input_mask.sum(-1))

        return (loglik- compensator)






