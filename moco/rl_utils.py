import distrax
import jax
import jax.numpy as jnp
from entmax_jax import entmax15
import chex
from chex import Array

def random_actor(logits, key):
    return distrax.Softmax(logits).sample(seed=key)

def greedy_actor(logits, key):
    return distrax.Greedy(logits).sample(seed=key)

def entmax_actor(logits, key):
	sparse_probs = entmax15(logits)
	return distrax.Categorical(probs=sparse_probs).sample(seed=key)

def entmax_policy_gradient_loss(
    logits_t: Array,
    a_t: Array,
    adv_t: Array,
    w_t: Array,
    use_stop_gradient: bool = True,
) -> Array:
	"""Calculates the policy gradient loss.

	See "Simple Gradient-Following Algorithms for Connectionist RL" by Williams.
	(http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)

	Args:
	logits_t: a sequence of unnormalized action preferences.
	a_t: a sequence of actions sampled from the preferences `logits_t`.
	adv_t: the observed or estimated advantages from executing actions `a_t`.
	w_t: a per timestep weighting for the loss.
	use_stop_gradient: bool indicating whether or not to apply stop gradient to
		advantages.

	Returns:
	Loss whose gradient corresponds to a policy gradient update.
	"""
	chex.assert_rank([logits_t, a_t, adv_t, w_t], [2, 1, 1, 1])
	chex.assert_type([logits_t, a_t, adv_t, w_t], [float, int, float, float])

	sparse_probs = entmax15(logits_t)
	log_pi_a_t = jnp.log(sparse_probs[jnp.arange(a_t.shape[0]),a_t])
	adv_t = jax.lax.select(use_stop_gradient, jax.lax.stop_gradient(adv_t), adv_t)
	loss_per_timestep = -log_pi_a_t * adv_t
	return jnp.mean(loss_per_timestep * w_t)

def rollout(key, problem, initial_position, params, model_fn, action_fn, env, rollout_length):

	def step_fn(state, key):
		logits = model_fn(params, state)
		action = action_fn(logits, key)
		new_state, timestep = env.step(state, action)
		return new_state, (logits, timestep)

	def run_n_step(state, key):
		random_keys = jax.random.split(key, rollout_length)
		state, rollout = jax.lax.scan(step_fn, state, random_keys)
		return state, rollout

	state, timestep = env.reset_from_problem(problem)
	state, timestep = env.step(state, initial_position)

	# Collect a rollout
	# key, subkey = jax.random.split(key)
	final_state, rollout = run_n_step(state, key)
	return final_state, rollout

def mis_rollout(key, problem, params, model_fn, action_fn, env, rollout_length):

	def step_fn(env_data, key):
		state, obs = env_data
		logits = model_fn(params, obs)
		action = action_fn(logits, key)
		new_state, new_obs = env.step(state, action)
		return (new_state, new_obs), (logits, state.is_done, action)

	def run_n_step(env_data, key):
		random_keys = jax.random.split(key, rollout_length)
		state, rollout = jax.lax.scan(step_fn, env_data, random_keys)
		return state, rollout

	env_data = env.reset_from_problem(problem)
	(final_state, final_obs), rollout = run_n_step(env_data, key)
	return final_state, rollout

def random_initial_position(key, problem_size):
	action = jax.random.randint(key, (), 0, problem_size)
	return action

def greedy_rollout(key, problem, params, model_fn, env):
	problem_size = problem.shape[0]
	final_state, (logits, timesteps) = rollout(key, problem, 0, params, model_fn, greedy_actor, env, problem_size-1)
	return final_state.trajectory, timesteps.reward.sum(-1)*-1

def pomo_rollout(key, problem, params, model_fn, env):
	problem_size = problem.shape[0]
	# make all possible starting positions
	start_positions = jnp.arange(problem_size)
	rollout_keys = jax.random.split(key, problem_size)
	batched_rollout = jax.vmap(rollout, in_axes=(0, None, 0, None, None, None, None, None))
	final_state, (logits, timesteps) = batched_rollout(rollout_keys, problem, start_positions, params, model_fn, greedy_actor, env, problem_size-1)

	total_rewards = timesteps.reward.sum(-1)
	best_sol_index = total_rewards.argmax()
	best_sol = final_state.trajectory[best_sol_index]
	return best_sol, total_rewards[best_sol_index]*-1