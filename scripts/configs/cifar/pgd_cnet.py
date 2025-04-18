STEP_SIZE = 1e-4
# Training Algorithm Config
algo_config = {
    'algorithm' : 'pgd',
    'n_particles' : 5,
    'theta_step_size' : STEP_SIZE,
    'q_step_size' :1e-3,
    'preconditioner': 'rmsprop',
}
