import torch 
from torch import Size, vmap
import numpy as np
from src.algorithms.algorithms import MomentumParticleAlgorithm 

 
class MPD_NC(MomentumParticleAlgorithm):
    '''
    No gradient correction scheme of MPD. With one partial update.
    '''
    def __init__(self,
                 model: torch.nn.Module,
                 theta_step_size: float,
                 q_step_size: float,
                 train_size: int,
                 eval_size: int,
                 n_particles: int,
                 device: str='cpu',
                 gamma_theta: float=3,
                 eta_theta: float=2,
                 gamma_x: float=0.5,
                 eta_x: float=10,
                 restart: bool=False,
                 **kwargs):
        super().__init__(model,
                         theta_step_size,
                         q_step_size,
                         n_particles,
                         train_size,
                         eval_size,
                         device,
                         eta_theta,
                         gamma_theta,
                         eta_x,
                         gamma_x,
                         catch_up=False,
                         preconditioner=None,
                         preconditioner_args=None)
        self.previous_loss = float('inf')
        self.restart = restart

    def reset_momentum(self):
        # Reset Momentum for Q.
        self.qmo = torch.randn_like(self.particles) / self.eta_x ** 0.5
        # Reset Momentum in Optimizer
        for name, param in self.model.named_parameters():
            self.tmo[name] = torch.zeros_like(param.data)

    def step(self, batch, batch_idx):
        t_dt = self.theta_step_size

        post = self.particles[batch_idx]
        
        tgameta = self.gamma_theta * self.eta_theta
        tomega = np.exp(- tgameta * t_dt)
        omtomega = 1 - tomega
        c1 = omtomega / self.gamma_theta

        loss = - self.model.log_prob(batch, post) / self.n_particles

        if self.restart and self.previous_loss < loss.item():
            self.reset_momentum()

        self.model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                tmo = self.tmo[name] 
                theta = param.data
                c1 = omtomega / self.gamma_theta
                c2 = (t_dt - omtomega / tgameta) / self.gamma_theta
                tmo_next = tomega * tmo - omtomega * param.grad.data / tgameta
                theta_next = theta + c1 * tmo - c2 * param.grad.data 

                # Update parameters
                param.data = theta_next
                self.tmo[name] = tmo_next

        self._step_q(batch, batch_idx)
        self.previous_loss = loss.item()
        return loss.item()


class MPD_NCTwo(MomentumParticleAlgorithm):
    '''
    No gradient correction scheme of MPD. Without two partial updates.
    '''
    def __init__(self,
                 model: torch.nn.Module,
                 theta_step_size: float,
                 q_step_size: float,
                 train_size: int,
                 eval_size: int,
                 n_particles: int,
                 device: str='cpu',
                 gamma_theta: float=3,
                 eta_theta: float=2,
                 gamma_x: float=0.5,
                 eta_x: float=10,
                 restart: bool=False,
                 **kwargs):
        super().__init__(model,
                         theta_step_size,
                         q_step_size,
                         n_particles,
                         train_size,
                         eval_size,
                         device,
                         eta_theta,
                         gamma_theta,
                         eta_x,
                         gamma_x,
                         catch_up=False,
                         preconditioner=None,
                         preconditioner_args=None)
        self.previous_loss = float('inf')
        self.restart = restart

    def reset_momentum(self):
        # Reset Momentum for Q.
        self.qmo = torch.randn_like(self.particles) / self.eta_x ** 0.5
        # Reset Momentum in Optimizer
        for name, param in self.model.named_parameters():
            self.tmo[name] = torch.zeros_like(param.data)

    def step(self, batch, batch_idx):
        t_dt = self.theta_step_size

        post = self.particles[batch_idx]
        
        tgameta = self.gamma_theta * self.eta_theta
        tomega = np.exp(- tgameta * t_dt)
        omtomega = 1 - tomega
        c1 = omtomega / self.gamma_theta

        loss = - self.model.log_prob(batch, post) / self.n_particles

        if self.restart and self.previous_loss < loss.item():
            self.reset_momentum()

        self.model.zero_grad()
        loss.backward()

        self._step_q(batch, batch_idx)

        # Theta step
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                tmo = self.tmo[name] 
                theta = param.data
                c1 = omtomega / self.gamma_theta
                c2 = (t_dt - omtomega / tgameta) / self.gamma_theta
                tmo_next = tomega * tmo - omtomega * param.grad.data / tgameta
                theta_next = theta + c1 * tmo - c2 * param.grad.data 

                # Update parameters
                param.data = theta_next
                self.tmo[name] = tmo_next
        self.previous_loss = loss.item()
        return loss.item()


class ParticleML_One(MomentumParticleAlgorithm):
    def __init__(self,
                 model: torch.nn.Module,
                 theta_step_size: float,
                 q_step_size: float,
                 train_size: int,
                 eval_size: int,
                 n_particles: int = 10,
                 device: str='cpu',
                 gamma_theta: float=0.9,
                 eta_theta: float=200,
                 gamma_x: float=5.,
                 eta_x: float=100,
                 restart: bool=False,
                 catch_up: bool=False,
                 preconditioner=None,
                 preconditioner_args={},
                 acc_theta=False,
                 acc_x=False,
                 **kwargs):
        super().__init__(model,
                         theta_step_size,
                         q_step_size,
                         n_particles,
                         train_size,
                         eval_size,
                         device,
                         eta_theta,
                         gamma_theta,
                         eta_x,
                         gamma_x,
                         catch_up=catch_up,
                         preconditioner=preconditioner,
                         preconditioner_args=preconditioner_args)
        # Initialize Optimizer
        self.previous_loss = float('inf')
        self.restart = restart
        self.acc_theta = acc_theta
        self.acc_x = acc_x
    
    def reset_momentum(self):
        # Reset Momentum for Q.
        self.qmo = torch.randn_like(self.particles) / self.eta_x ** 0.5
        # Reset Momentum in Optimizer
        for name, param in self.model.named_parameters():
            self.tmo[name] = torch.zeros_like(param.data)

    def _theta_step(self, batch, batch_idx):
        self.model.train()
        t_dt = self.theta_step_size
        post = self.particles[batch_idx]

        def _backward_theta():
            if self.theta_step_size > 0:
                loss = - self.model.log_prob(batch, post) / self.n_particles 
                self.model.zero_grad()
                loss.backward()
            if self.preconditioner is not None:
                self.preconditioner.step()
            return loss

        if self.acc_theta:
            tgameta = self.gamma_theta * self.eta_theta
            tomega = np.exp(- tgameta * t_dt)
            omtomega = 1 - tomega
            c1 = omtomega / self.gamma_theta

            for name, param in self.model.named_parameters():
                tmo = self.tmo[name]
                param.data = param.data + c1 * tmo
            
            loss = _backward_theta()

            with torch.no_grad():
                # Update parameters
                if self.theta_step_size > 0:
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            tmo = self.tmo[name] 
                            theta = param.data
                            c1 = omtomega / self.gamma_theta
                            c2 = (t_dt - omtomega / tgameta) / self.gamma_theta
                            tmo_next = tomega * tmo - omtomega * param.grad.data / tgameta
                            theta_next = theta - c2 * param.grad.data 

                            # Update parameters
                            param.data = theta_next
                            self.tmo[name] = tmo_next
        else:
            loss = _backward_theta()

            # Theta step
            if self.theta_step_size > 0:
                for param in self.model.parameters():
                    if param.requires_grad and param.grad is not None:
                        param.data = param.data - self.theta_step_size * param.grad.data
        self.previous_loss = loss.item()
        return  loss.item()

    def _q_step(self, batch, batch_idx):
        self.model.eval()
        if self.acc_x:
            self._step_q(batch, batch_idx)
        else:
            dt = self.q_step_size
            with torch.no_grad():
                post = self.particles[batch_idx]
                with torch.enable_grad():
                    prev_x = post.requires_grad_(True)
                    log_p_v = self.model.log_prob(batch, prev_x)
                    score = torch.autograd.grad(log_p_v, prev_x)[0].detach().clone()
                    prev_x = prev_x.requires_grad_(False)

            noise = torch.randn_like(post)
            self.particles[batch_idx] += (score * dt + noise * (2 * dt) ** 0.5).detach().clone()

    def _step(self, batch, batch_idx):
        loss = self._theta_step(batch, batch_idx)

        # Gradient step particles
        self._q_step(batch, batch_idx)

        if self.catch_up:
            missed_idx = np.full(len(self.missed), True, dtype=bool)
            missed_idx[batch_idx] = False
            self.missed[missed_idx] += 1
        return loss