params_hmc = hamiltorch.sample(log_prob_func=log_prob_func,
							   params_init=params_init,
							   num_samples=num_samples,
							   step_size=step_size,
							   num_steps_per_sample=num_steps_per_sample)
