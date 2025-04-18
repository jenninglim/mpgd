def choose_eta_theta(gamma_theta,
                     theta_step_size,
                     momentum=0.9):
    gamma_theta = (1 - momentum) / gamma_theta / theta_step_size
    return gamma_theta


STEP_SIZE = 1e-4
GAMMA = 0.9
# Training Algorithm Config
algo_config = {
    'algorithm' : 'MPD',
    'n_particles' : 5,
    'theta_step_size' : STEP_SIZE,
    'gamma_theta' : GAMMA,
    'q_step_size' : STEP_SIZE,
    'catch_up' : True,
    'gamma_x' : GAMMA,
    'eta_theta' : choose_eta_theta(GAMMA, STEP_SIZE, momentum=0.95),
    'eta_x' : choose_eta_theta(GAMMA, STEP_SIZE, momentum=0.0),
    'preconditioner': 'rmsprop',
    'restart': False,
}
