STEP_SIZE = 1e-4
# Training Algorithm Config
algo_config = {
    'algorithm' : 'sr',
    'theta_step_size' : STEP_SIZE,
    'q_step_size' :1e-3,
    'preconditioner': 'rmsprop',
    'n_chain_length' : 20,
}
