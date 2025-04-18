def choose_eta_theta(gamma_theta,
                     theta_step_size,
                     momentum=0.9):
    gamma_theta = (1 - momentum) / gamma_theta / theta_step_size
    return gamma_theta

STEP_SIZE = 1e-4
GAMMA = 0.9 # 0.9
# Training Algorithm Config
algo_config = {
    'algorithm' : 'MPD',
    'n_particles' : 5,
    'theta_step_size' : STEP_SIZE,
    'gamma_theta' : GAMMA,
    'q_step_size' : STEP_SIZE,
    'gamma_x' : GAMMA,
    'eta_theta' : 500, #  choose_eta_theta(GAMMA, STEP_SIZE, momentum=0.7),
    'eta_x' : 500, #  choose_eta_theta(GAMMA, STEP_SIZE, momentum=0.5),
    'catch_up' : True,
    'preconditioner': 'rmsprop',
    'restart': False,
}
