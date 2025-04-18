from torch.utils.data import DataLoader
from src.algorithms.algorithms import *
from src.preconditioners import *
import math
import numpy as np


class Algorithm:
    def __init__(self,
                 model,
                 theta_step_size,
                 q_step_size,
                 device,
                 preconditioner=None,
                 preconditioner_args={}):
        self.model = model
        self.theta_step_size = theta_step_size
        self.q_step_size = q_step_size
        self.device = device
        self.preconditioner = None
        if preconditioner is not None:
            if preconditioner in PRECONDITIONERS.keys():
                self.preconditioner = PRECONDITIONERS[preconditioner](model,
                                                                    **preconditioner_args)
            else:
                self.preconditioner = preconditioner
                                                            
    
    def step(self, batch, batch_idx):
        raise NotImplementedError()
    

    def _step_q(self, batch, batch_idx):
        print("Warning: _step_q not implemented.")


class ParticleAlgorithm(Algorithm):
    def __init__(self,
                 model,
                 theta_step_size,
                 q_step_size,
                 n_particles,
                 train_size,
                 eval_size,
                 device,
                 catch_up=False,
                 preconditioner=None,
                 preconditioner_args={}):
        super().__init__(model,
                         theta_step_size,
                         q_step_size,
                         device,
                         preconditioner=preconditioner,
                         preconditioner_args=preconditioner_args)
        self.train_size = train_size
        self.eval_size = eval_size
        self.n_particles = n_particles
        batch_size = 512
        K = math.ceil((train_size + eval_size) / batch_size)
        particles = torch.randn(train_size + eval_size,
                                n_particles,
                                model.latent_size()).to(device)
        self.particles = particles
        self.catch_up = catch_up
        if catch_up is True:
            self.missed = np.zeros(train_size + eval_size, dtype=int)

    def reset_particles(self, idx):
        if hasattr(self, "qmo"):
            self.qmo[idx] = torch.randn_like(self.particles[idx]) / self.eta_x ** 0.5
        self.particles[idx] = torch.randn_like(self.particles[idx])
    
    def _eval_loss(self, batch, idx):
        post = self.particles[idx]
        loss =- self.model.log_prob(batch, post).item() / self.n_particles
        return loss 

    def step(self, batch, batch_idx):
        self._catchup_particles(batch, batch_idx)
        loss = self._step(batch, batch_idx)
        assert not np.isnan(loss), "Loss is NaN."
        return loss

    def _catchup_particles(self, batch, batch_idx):
        if self.catch_up:
            missed = self.missed[batch_idx]
            update_idx = missed > 0
            if update_idx.sum() > 0:
                self._step_q(
                    batch[update_idx],
                    batch_idx[update_idx],
                    step_size=self.q_step_size * missed[update_idx]
                    )
            self.missed[batch_idx] = 0

    def _step(self, batch, batch_idx):
        raise NotImplementedError()


class MomentumParticleAlgorithm(ParticleAlgorithm):
    def __init__(self,
                 model,
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
                 preconditioner_args={}):
        # Momentum for Q.
        super().__init__(model,
                         theta_step_size,
                         q_step_size,
                         n_particles,
                         train_size,
                         eval_size,
                         device,
                         catch_up=catch_up,
                         preconditioner=preconditioner,
                         preconditioner_args=preconditioner_args)
        self.qmo = torch.randn_like(self.particles) / eta_x ** 0.5
        # Momentum for Theta.
        self.tmo = {}
        for name, param in model.named_parameters():
            self.tmo[name] = torch.zeros_like(param.data)
        self.gamma_x = gamma_x
        self.eta_x = eta_x
        self.gamma_theta = gamma_theta
        self.eta_theta = eta_theta
    

    def _step_q(self, batch, batch_idx, step_size=None):
        if step_size is None:
            step_size = self.q_step_size
        post = self.particles[batch_idx]
        qmo = self.qmo[batch_idx]
        with torch.no_grad():
            next_post, next_qmo = transition(self.model,
                                             batch,
                                             self.gamma_x,
                                             self.eta_x,
                                             step_size,
                                             post,
                                             qmo)

            self.particles[batch_idx] = next_post.detach().clone()
            self.qmo[batch_idx] = next_qmo.detach().clone()


class ParticleML(ParticleAlgorithm):
    def __init__(self,
                 model: torch.nn.Module,
                 theta_step_size: float,
                 q_step_size: float,
                 train_size: int,
                 eval_size: int,
                 device: str='cpu',
                 n_particles: int=10,
                 preconditioner = None,
                 catch_up: bool=False,
                 preconditioner_args={},
                 **kwargs):
        super().__init__(model,
                         theta_step_size,
                         q_step_size,
                         n_particles,
                         train_size,
                         eval_size,
                         device,
                         catch_up=catch_up,
                         preconditioner=preconditioner,
                         preconditioner_args=preconditioner_args)

    def _step(self, batch, batch_idx):
        """
        batch: Size(batch_size, dim)
        """
        post = self.particles[batch_idx]
        loss = - self.model.log_prob(batch, post) / self.n_particles

        self.model.zero_grad()
        loss.backward()
        if self.preconditioner is not None:
            self.preconditioner.step()
        
        # Theta step
        if self.q_step_size > 0:
            self._step_q(batch, batch_idx)
        if self.theta_step_size > 0:
            for param in self.model.parameters():
                if param.requires_grad and param.grad is not None:
                    param.data = param.data - self.theta_step_size * param.grad.data
        return loss.item()
    
    def _step_q(self, batch, batch_idx, step_size=None):
        dt = step_size if step_size is not None else self.q_step_size
        with torch.no_grad():
            post = self.particles[batch_idx]
            with torch.enable_grad():
                prev_x = post.requires_grad_(True)
                log_p_v = self.model.log_prob(batch, prev_x)
                score = torch.autograd.grad(log_p_v, prev_x)[0].detach().clone()
                prev_x = prev_x.requires_grad_(False)

            noise = torch.randn_like(post)
            self.particles[batch_idx] += (score * dt + noise * (2 * dt) ** 0.5).detach().clone()


class MPD_Nesterov(MomentumParticleAlgorithm):
    '''
    Accelerated partiel ML with Nesterov Discretization.
    '''
    def __init__(self,
                 model: torch.nn.Module,
                 theta_step_size: float,
                 q_step_size: float,
                 n_particles: int,
                 train_size: int,
                 eval_size: int,
                 device: str='cpu',
                 gamma_x: float=0.5,
                 gamma_theta: float=0.9,
                 eta_theta: float=200,
                 eta_x: float=100,
                 restart: bool=False,
                 catch_up: bool=False,
                 **kwargs):
        super().__init__(model,
                         theta_step_size,
                         q_step_size,
                         n_particles,
                         train_size,
                         eval_size,
                         device,
                         None,
                         None,
                         eta_x,
                         gamma_x,
                         catch_up=False)
        self.tmo_param = 1 - theta_step_size * gamma_theta * eta_theta
        # assert self.tmo_param > 0 and self.tmo_param < 1, "Theta momentum parameter out of range."
        self.model = model
        self.restart = restart

    def reset_momentum(self):
        # Reset Momentum for Q.
        self.qmo = torch.randn_like(self.particles) / self.eta_x ** 0.5
        # Reset Momentum in Optimizer
        self.tmo = torch.zeros_like(self.particles) 

    def _step(self, batch, batch_idx):
        self.model.train()
        d = self.particles.shape[-1]

        post = self.particles[batch_idx]
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param._grad is not None:
                    param.data = param.data + self.tmo_param * self.tmo[name]

        loss = - self.model.log_prob(batch, post) / self.n_particles

        if self.restart and self.previous_loss < loss.item():
            self.reset_momentum()

        self.model.zero_grad()
        loss.backward()
        dt_squared = self.theta_step_size ** 2
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param._grad is not None:
                    param.data = param.data + self.tmo[name]
                    self.tmo[name] = self.tmo_param * self.tmo[name] - dt_squared * param._grad.data
        self.model.eval()

        # Gradient step particles
        self._step_q(batch, batch_idx)
        self.previous_loss = loss.item()
        return  loss.item()


class MPD_Proposed(MomentumParticleAlgorithm):
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
        # assert self.tmo_param > 0 and self.tmo_param < 1, "Theta momentum parameter out of range."
        self.previous_loss = float('inf')
        self.restart = restart
    
    def reset_momentum(self):
        # Reset Momentum for Q.
        self.qmo = torch.randn_like(self.particles) / self.eta_x ** 0.5
        # Reset Momentum in Optimizer
        for name, param in self.model.named_parameters():
            self.tmo[name] = torch.zeros_like(param.data)

    def _step(self, batch, batch_idx):
        self.model.train()
        t_dt = self.theta_step_size
        post = self.particles[batch_idx]
        tgameta = self.gamma_theta * self.eta_theta
        tomega = np.exp(- tgameta * t_dt)
        omtomega = 1 - tomega
        c1 = omtomega / self.gamma_theta

        for name, param in self.model.named_parameters():
            tmo = self.tmo[name]
            param.data = param.data + c1 * tmo

        if self.theta_step_size > 0:
            loss = - self.model.log_prob(batch, post) / self.n_particles

            self.model.zero_grad()
            loss.backward()

        if self.preconditioner is not None:
            self.preconditioner.step()

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

        # Gradient step particles
        self.model.eval()
        if self.q_step_size > 0:
            self._step_q(batch, batch_idx)

        if self.catch_up:
            missed_idx = np.full(len(self.missed), True, dtype=bool)
            missed_idx[batch_idx] = False
            self.missed[missed_idx] += 1
        self.previous_loss = loss.item()
        return  loss.item()


class ShortRun(ParticleAlgorithm):
    """
    Implementation of Short Run algorithm
    (see https://arxiv.org/abs/1912.01909).
    """
    def __init__(self,
                 model,
                 theta_step_size,
                 q_step_size,
                 train_size,
                 eval_size,
                 device,
                 preconditioner=None,
                 n_chain_length=25,
                 **kwargs):
        super().__init__(model,
                         theta_step_size=theta_step_size,
                         q_step_size=q_step_size,
                         n_particles=1,
                         train_size=train_size,
                         eval_size=eval_size,
                         preconditioner=preconditioner,
                         device=device)
        self.n_chain_length = n_chain_length
        self.q_step_size = q_step_size

    def _step(self, img_batch, idx):
        """
        Runs a short chain (with length self.n_chain_length) Langevin
        algorithm with step size self.particle_step_size.
        Note that the chains are initialized randomly.
        """
        # Run short run
        self.model.eval()
        self.model.requires_grad_(False)

        # Initialise particles:
        particles = self.model.prior.sample(img_batch.shape[0] * self.n_particles)
        particles = particles.reshape(img_batch.shape[0],
                                      self.n_particles,
                                      self.model.latent_size())
        particles = particles.to(img_batch.device).requires_grad_(True)
        assert not particles.isnan().any(), f"found {particles.isnan().sum()} NaNs in particles"

        # Run chain:
        for i in range(self.n_chain_length):
            log_p_v = self.model.log_prob(img_batch,
                                          particles).sum()
            x_grad = torch.autograd.grad(log_p_v, particles
                                         )[0].detach().clone()

            particles = particles + (self.q_step_size
                                     * x_grad.to(particles.device))
            particles = particles + ((2 * self.q_step_size) ** 0.5
                                     * torch.randn_like(particles))
            particles = particles.detach().clone().requires_grad_(True)

        # Turn on theta gradients:
        self.model.train()
        self.model.requires_grad_(True)

        # Evaluate loss function:
        loss = - self.model.log_prob(img_batch, particles) / self.n_particles

        # Backpropagate theta gradients:
        self.model.zero_grad()
        loss.backward()

        if self.preconditioner is not None:
            self.preconditioner.step()

        # Update theta:
        for param in self.model.parameters():
            param.data = param.data - self.theta_step_size * param.grad.data

        # Return value of loss function:
        return loss.item()


class ABP(ParticleAlgorithm):
    """
    Implementation of a sub-sampled version of Alternating Backprop: a
    persistent version of short run algorithm.
    See https://arxiv.org/abs/1606.08571.
    """
    def __init__(self,
                 model,
                 theta_step_size: float = 1e-3,
                 q_step_size: float = 0.1,
                 train_size: int = 100,
                 eval_size: int = 10,
                 n_chain_length: int = 25,
                 device: str = 'cpu',
                 preconditioner=None,
                 **kwargs):
        super().__init__(model,
                         theta_step_size=theta_step_size,
                         q_step_size=q_step_size,
                         n_particles=1,
                         train_size=train_size,
                         eval_size=eval_size,
                         preconditioner=preconditioner,
                         device=device)
        self.n_chain_length = n_chain_length

    def _step(self,
              img_batch,
              idx):
        """
        Runs a short chain (with length self.n_chain_length) Langevin algorithm
        with step size self.particle_step_size.
        The chain is initialized from its the previous step.
        """
        # Run short run
        self.model.eval()

        # Run chain for the subset of the particle cloud.
        particles = self.particles[idx].detach().clone().to(self.device)\
            .requires_grad_(True)
        for i in range(self.n_chain_length):
            log_p_v = self.model.log_prob(img_batch.to(self.device),
                                          particles).sum()
            x_grad = torch.autograd.grad(log_p_v, particles
                                         )[0].detach().clone()

            particles = particles + (self.q_step_size
                                     * x_grad.to(particles.device))
            particles = particles + ((2 * self.q_step_size) ** 0.5
                                     * torch.randn_like(particles))
            particles = particles.detach().clone().requires_grad_(True)
        # Update posterior.
        self.particles[idx] = particles.detach().clone()\
            .to(self.particles.device)
        # Compute theta gradients

        # Turn on theta gradients:
        self.model.train()
        self.model.requires_grad_(True)

        # Evaluate loss function:
        loss = - self.model.log_prob(img_batch,
                                     self.particles[idx]) / self.n_particles

        # Backpropagate theta gradients:
        self.model.zero_grad()
        loss.backward()

        if self.preconditioner is not None:
            self.preconditioner.step()

        # Update theta:
        for param in self.model.parameters():
            param.data = param.data - self.theta_step_size * param.grad.data
        self._posterior_up_to_date = False

        # Return value of loss function:
        return loss.item()


def transition(model,
               y,
               gamma_x,
               eta_x,
               q_dt,
               prev_x,
               prev_qmo):
        '''
        Transition kernel for the particles.
        '''
        DEVICE = prev_x.device
        DTYPE = prev_x.dtype
        with torch.enable_grad():
            prev_x = prev_x.requires_grad_(True)
            log_p_v = model.log_prob(y, prev_x)
            score = torch.autograd.grad(log_p_v, prev_x)[0].detach().clone()
            prev_x = prev_x.requires_grad_(False)

        d = prev_x.shape[-1]
        gameta = gamma_x * eta_x
        scale = np.exp(- gameta * q_dt) 
        scale2 = np.exp(- 2 * gameta * q_dt) 

        ## Covariance matrix.
        s_XX = 1 /  gamma_x * (2 * q_dt - scale2 / gameta + 4 * scale/ gameta -  3 /gameta)
        s_mm = (1 - scale2) / eta_x
        s_mX = 1 /  gameta * (1 - 2 * scale + scale2)

        ## Cholesky Decomposition.
        L_XX = s_XX ** 0.5
        L_mX = s_mX / (s_XX ** 0.5)
        L_mm = (s_mm - s_mX ** 2/ s_XX) ** 0.5

        to_tensor = lambda x: torch.tensor(x, dtype=DTYPE, device=DEVICE).unsqueeze(-1).unsqueeze(-1)
        scale = to_tensor(scale)
        scale2 = to_tensor(scale2)
        q_dt = to_tensor(q_dt)
        L_XX = to_tensor(L_XX)
        L_mX = to_tensor(L_mX)
        L_mm = to_tensor(L_mm)

        noise = torch.randn(*prev_x.shape[:-1], 2 * d, device=DEVICE, dtype=DTYPE)
        post_noise = L_XX * noise[..., :d]
        m_noise = L_mX * noise[..., :d] + L_mm * noise[..., d:]
        next_post = prev_x + 1 / gamma_x * ((1 -  scale) * prev_qmo + (q_dt - (1- scale)/ gameta ) * score) + post_noise
        next_qmo = scale * prev_qmo + (1- scale) / gameta * score + m_noise
        assert next_post.shape == prev_x.shape
        assert next_qmo.shape == prev_qmo.shape
        return next_post, next_qmo
