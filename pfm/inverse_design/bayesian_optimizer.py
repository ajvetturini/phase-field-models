""" Perform Bayesian optimization to find the best parameters for a given objective function. """
from typing import Dict, List, Tuple, Any, Callable
import abc
import jax
import jax.numpy as jnp
from functools import partial
from dataclasses import dataclass
from pathlib import Path
import toml
import matplotlib.pyplot as plt
import numpy as np

try:
    import bayex
    import optax
except ImportError:
    raise ImportError("ERROR: Bayex not installed, use pip to install before using this function")

class BOObjective(abc.ABC):
    """
    Abstract base class for Bayesian Optimization objectives

    This class serves as an interface. To use it, a user must subclass it and
    implement their own `forward_simulation` and `loss_function` methods
    tailored to their specific scientific problem
    """
    def __init__(self, static_sim_data: Dict[str, Any], float_type='float32'):
        """Initializes the objective and JIT-compiles the evaluation functions."""
        self.static_sim_data = static_sim_data
        loss_fn_with_static_data = partial(self.loss_function, static_data=self.static_sim_data)

        # JIT-compile the function for a single evaluation.
        self._value_and_grad_fn = jax.jit(
            jax.value_and_grad(loss_fn_with_static_data)
        )
        self._vmapped_value_and_grad_fn = jax.jit(
            jax.vmap(jax.value_and_grad(loss_fn_with_static_data))
        )

        self._value_no_grad_fn = jax.jit(
            loss_fn_with_static_data
        )
        self._vmapped_value_no_grad_fn = jax.jit(
            jax.vmap(loss_fn_with_static_data)
        )

        self._float_dtype = jnp.float64 if (float_type.lower() == 'float64') else jnp.float32

    @abc.abstractmethod
    def forward_simulation(self, params: Dict[str, jnp.ndarray], static_data: Dict[str, Any]) -> jnp.ndarray:
        """
        The user's core "expensive" simulation logic goes here.

        Args:
            params: A dictionary mapping parameter names (e.g., 'epsilon', 'deltaH')
                    to their JAX array values.

        Returns:
            A JAX array representing the simulation's output feature to be targeted.
        """
        pass

    @abc.abstractmethod
    def final_forward_simulation(self, params: Dict[str, jnp.ndarray], static_data: Dict[str, Any]) -> jnp.ndarray:
        """ Final simulation that will WRITE the export files """
        pass

    @abc.abstractmethod
    def loss_function(self, params: Dict[str, jnp.ndarray], static_data: Dict[str, Any]) -> jnp.ndarray:
        """
        Calculates the scalar loss based on the simulation output.

        Args:
            params: A dictionary of parameters to be passed to the simulation.

        Returns:
            A scalar JAX array representing the loss.
        """
        pass

    def evaluate_with_grads(self, params: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Evaluates the loss and gradient for a single set of parameters.
        This is the primary interface for a standard, sequential BO loop.

        NOTE: This becomes VERY memory intensive!

        Args:
            params: A dictionary of standard Python floats for the parameters

        Returns:
            A tuple of (loss, gradient_dict).
        """
        params_jnp = {k: jnp.array(v) for k, v in params.items()}
        loss, grad_jnp = self._value_and_grad_fn(params_jnp, compute_grad=False)
        grad_py = {k: v.item() for k, v in grad_jnp.items()}
        return loss.item(), grad_py

    def evaluate_batch_with_grads(self, params_batch: List[Dict[str, float]]) -> Tuple[List[float], List[Dict[str, float]]]:
        """
        Evaluates a batch of parameter sets in parallel using vmap.
        This is the interface for a parallelized BO scheduler.

        NOTE: This becomes VERY memory intensive!

        Args:
            params_batch: A list of parameter dictionaries.

        Returns:
            A tuple of (list_of_losses, list_of_gradient_dicts).
        """
        if not params_batch:
            return [], []
        pytree = {k: jnp.array([d[k] for d in params_batch]) for k in params_batch[0]}
        losses_jnp, grads_jnp = self._vmapped_value_and_grad_fn(pytree)
        num_items = len(params_batch)
        grads_py = [{k: v[i].item() for k, v in grads_jnp.items()} for i in range(num_items)]
        return losses_jnp.tolist(), grads_py

    def evaluate_without_grads(self, params: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Evaluates the loss and gradient for a single set of parameters.
        This is the primary interface for a standard, sequential BO loop.

        Args:
            params: A dictionary of standard Python floats for the parameters

        Returns:
            A tuple of (loss, {}).
        """
        params_jnp = {k: jnp.array(v) for k, v in params.items()}
        loss = self._value_no_grad_fn(params_jnp)
        return loss.item(), {}

    def evaluate_batch_without_grads(self, params_batch: List[Dict[str, float]]) -> Tuple[
        List[float], List[Dict[str, float]]]:
        """
        Evaluates a batch of parameter sets in parallel using vmap.
        This is the interface for a parallelized BO scheduler.

        Args:
            params_batch: A list of parameter dictionaries.

        Returns:
            A tuple of (list_of_losses, {}).
        """
        if not params_batch:
            return [], []
        pytree = {k: jnp.array([d[k] for d in params_batch]) for k in params_batch[0]}
        losses_jnp = self._vmapped_value_no_grad_fn(pytree)
        return losses_jnp.tolist(), {}


@dataclass
class BOArgs:
    """
    Arguments for running Bayesian Optimization using bayex
    """
    write_location: Path | str = ''  # Default directory is wherever user runs script from
    num_iterations: int = 20
    num_initial_points: int = 5
    batch_size: int = 4  # Number of simulations to run in parallel per iteration
    parameter_bounds: Dict[str, Tuple[float, float]] = None  # {Parameter1: (lb1, ub1), Parameter2: (lb2, ub2), ...}
    verbose: bool = False
    maximize: bool = False
    acquisition_fn: str = 'PI'
    seed: int = 8

def run_bayesian_optimization(objective: BOObjective, args: BOArgs, visualize_gp: bool = False):
    """ Runs a simple Bayesian Optimization loop using the provided objective and arguments. """
    final_location = Path(args.write_location) / 'results'
    final_location.mkdir(parents=True, exist_ok=True)  # Create a results folder for storing results

    ori_key = jax.random.key(args.seed)
    ini_key, loop_key = jax.random.split(ori_key)
    acq_fn = args.acquisition_fn.upper()

    if acq_fn not in ['EI', 'PI', 'UCB', 'LCB']:
        raise ValueError("ERROR: Acquisition function must be one of `EI`, `PI`, `UCB`, `LCB` for use with bayex.")
    if args.parameter_bounds is None:
        raise ValueError("ERROR: Parameter bounds must be specified in BOArgs.")

    # Initialize the optimizer with the specified bounds
    param_names = list(args.parameter_bounds.keys())
    bounds = jnp.array([args.parameter_bounds[name] for name in param_names])  # shape (n_params, 2)
    lows, highs = bounds[:, 0], bounds[:, 1]

    # Setup design space / bayex optimizer
    space = {name: bayex.domain.Real(low, high) for name, (low, high) in args.parameter_bounds.items()}
    optimizer = bayex.Optimizer(
        domain=space,
        maximize=args.maximize,
        acq=acq_fn
    )

    # Get some prior evaluations to initialize GP
    ini_key, subkey = jax.random.split(ini_key)
    initial_points = jax.random.uniform(
        subkey, shape=(args.num_initial_points, len(param_names)), minval=lows, maxval=highs
    )
    initial_points_dicts = [dict(zip(param_names, p)) for p in initial_points]  # Formatted for evaluator
    params = {i: [] for i in param_names}
    for d in initial_points_dicts:
        for k, v in d.items():
            params[k].append(float(v))

    # Batch out initial_points_dicts and evaluate:
    batches = [initial_points_dicts[i:i + args.batch_size] for i in range(0, len(initial_points_dicts), args.batch_size)]
    all_losses = []
    for b in batches:
        initial_losses, _ = objective.evaluate_batch_without_grads(b)
        all_losses.extend(initial_losses)

    opt_state = optimizer.init(all_losses, params)

    for i in range(args.num_iterations):
        if args.verbose:
            print(f'Beginning iteration {i}')
        key = jax.random.fold_in(loop_key, i)
        new_params = optimizer.sample(key, opt_state)
        losses, _ = objective.evaluate_without_grads(new_params)
        opt_state = optimizer.fit(opt_state, losses, new_params)

        # Note: Look to use something like BoTorch to use the gradient information instead, right now scikit-optimize
        # does not use this information natively
        if args.verbose:
            if isinstance(losses, list):
                print(f"  - Observed losses at {i}: {[f'{l:.4f}' for l in losses]}")
            else:
                print(f"  - Observed loss at {i}: {losses}")

    # Finally, get the best results for returning:
    best_result = opt_state.best_score
    best_params_dict = opt_state.best_params

    if args.verbose:
        print(f"Minimum loss found: {best_result:.6f}")
        print(f"Optimal parameters: {best_params_dict}")

    final_sim_eval_value, final_toml = objective.final_forward_simulation(
        {k: jnp.array(v) for k, v in best_params_dict.items()},
        objective.static_sim_data
    )

    output_toml = final_location / 'optimized_input_file.toml'
    with open(output_toml, "w") as f:
        toml.dump(final_toml, f)

    if args.verbose:
        print(f"Bayesian Optimization finished...")
        print(f"  - Predicted eval at optimum: {final_sim_eval_value.item():.4f}")

    _visualize_gp(opt_state, args, final_location, visualize_gp)
    return best_result, best_params_dict

def _visualize_gp(opt_state, args: BOArgs, final_location: Path, show_plot: bool = False):
    # Visualize the gaussian process, relies on the bayex optimizer (not compatible with skopt)
    param_name = list(args.parameter_bounds.keys())
    from bayex.gp import gaussian_process

    for p in param_name:
        # Collect relevent data:
        bounds = args.parameter_bounds[p]
        x_plot = jnp.linspace(bounds[0], bounds[1], 200).reshape(-1, 1)
        xs = opt_state.params[p]
        xs = xs[:, None]
        mean, std = gaussian_process(opt_state.gp_state.params, xs, opt_state.ys, opt_state.mask, xt=x_plot)

        # Create Plot
        mean = np.array(mean)
        std = np.array(std)
        plt.figure(figsize=(10, 6))
        plt.plot(x_plot, mean, 'b-', label='GP Mean (Estimate)')
        plt.fill_between(x_plot.flatten(), mean - 1.96 * std, mean + 1.96 * std,
                         color='blue', alpha=0.15, label='95% Confidence Interval')

        x_observed, y_observed = np.array(opt_state.params[p]), np.array(opt_state.ys)
        mask = np.array(opt_state.mask)
        x_observed = x_observed[mask]
        y_observed = y_observed[mask]
        plt.plot(x_observed, y_observed, 'ko', markersize=8, label='Observations')


        if args.maximize:
            best_x = x_observed[np.argmax(y_observed)]
            best_y = np.max(y_observed)
        else:
            best_x = x_observed[np.argmin(y_observed)]
            best_y = np.min(y_observed)

        plt.plot(best_x, best_y, 'y*', markersize=15, markeredgecolor='k', label=f'Best Found: {best_x:.2f}')

        plt.title('Bayesian Optimization of Epsilon Parameter')
        plt.xlabel(f'{p}')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--')
        plt.tight_layout()
        if show_plot:
            plt.show()
        op = final_location / f'bayes_opt_{p}_results.png'
        plt.savefig(op, dpi=400)
        plt.close()


def _optimize_suggestion(params: dict, fun: Callable, max_iter: int = 10):
    """
    Applies local optimization (L-BFGS) to a given starting point.

    This function refines candidate points proposed by the acquisition
    function by performing a fixed number of gradient-based optimization
    steps using Optax's L-BFGS optimizer.

    Args:
        params (jax.Array): Initial point in input space to optimize.
        fun (Callable): Objective function to maximize. It must return a
            scalar value and support automatic differentiation.
        max_iter (int): Maximum number of L-BFGS steps to apply.

    Returns:
        jax.Array: The optimized point after `max_iter` iterations.
    """

    # L-BFGS optimizer is used for minimization, so to maximize acquisition
    # we need to negate the function value.
    opt = optax.lbfgs()
    value_and_grad_fun = optax.value_and_grad_from_state(lambda x: -fun(x))

    def step(carry, _):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(grad, state, params,
                                    value=value, grad=grad, value_fn=fun)
        params = optax.apply_updates(params, updates)
        params = jnp.clip(params, -1e6, 1e6)
        return (params, state), None

    init_carry = (params, opt.init(params))
    (final_params, _), __ = jax.lax.scan(step, init_carry, None, length=max_iter)
    return final_params


class ParallelOptimizer(bayex.Optimizer):

    def __init__(self, domain: dict, acq: str = 'EI', maximize: bool = False):
        super().__init__(domain, acq, maximize)

    @partial(jax.jit, static_argnames=('self', 'num_samples', 'size'))
    def sample(self, key, opt_state, num_samples, size=10_000, has_prior=False):
        """
        Samples new parameters using the acquisition function. This is a parrellized version

        Args:
            key: JAX PseudoRandom key for random sampling.
            opt_state: Current optimizer state.
            size: Number of samples to draw.
            has_prior: If True, also return GP predictions.

        Returns:
            Sampled parameters (dict), and optionally (xs_samples, means, stds).
        """
        # Sample 'size' elements of each distribution.
        keys = jax.random.split(key, len(opt_state.params))
        samples = {param: self.domain[param].sample(key, (size,))
                   for key, param in zip(keys, opt_state.params)}

        xs = jnp.stack([self.domain[key].transform(opt_state.params[key])
                        for key in opt_state.params], axis=1)
        ys = opt_state.ys
        mask = opt_state.mask
        gpparams = opt_state.gp_state.params
        keys = jax.random.split(key, len(opt_state.params))
        xs_samples = jnp.stack([self.domain[name].sample(key, (size,))
                                for key, name in zip(keys, opt_state.params)], axis=1)

        # Use the acquisition function to find the best parameters
        zs, (means, stds) = self.acq(xs_samples, xs, ys, mask, gpparams)

        # For parallel sampling, we will simply take the greediest top num_samples
        top_vals, top_idx = jax.lax.top_k(zs, num_samples)
        top_params = jax.tree_util.tree_map(lambda d: d[top_idx], samples)
        if has_prior:
            return top_params, (xs_samples, means, stds)
        return top_params

def run_parallel_bayesian_optimization(objective: BOObjective, args: BOArgs, visualize_gp: bool = False):
    """ Runs a parralellized version of Bayesian Optimization loop using the provided objective and arguments.
    This uses a CUSTOM sampler described in the above methodology
    """
    final_location = Path(args.write_location) / 'results'
    final_location.mkdir(parents=True, exist_ok=True)  # Create a results folder for storing results

    ori_key = jax.random.key(args.seed)
    ini_key, loop_key = jax.random.split(ori_key)
    acq_fn = args.acquisition_fn.upper()

    if acq_fn not in ['EI', 'PI', 'UCB', 'LCB']:
        raise ValueError("ERROR: Acquisition function must be one of `EI`, `PI`, `UCB`, `LCB` for use with bayex.")
    if args.parameter_bounds is None:
        raise ValueError("ERROR: Parameter bounds must be specified in BOArgs.")

    # Initialize the optimizer with the specified bounds
    param_names = list(args.parameter_bounds.keys())
    bounds = jnp.array([args.parameter_bounds[name] for name in param_names])  # shape (n_params, 2)
    lows, highs = bounds[:, 0], bounds[:, 1]

    # Setup design space / bayex optimizer
    space = {name: bayex.domain.Real(low, high) for name, (low, high) in args.parameter_bounds.items()}
    optimizer = ParallelOptimizer(
        domain=space,
        maximize=args.maximize,
        acq=acq_fn
    )

    # Get some prior evaluations to initialize GP
    ini_key, subkey = jax.random.split(ini_key)
    initial_points = jax.random.uniform(
        subkey, shape=(args.num_initial_points, len(param_names)), minval=lows, maxval=highs
    )
    initial_points_dicts = [dict(zip(param_names, p)) for p in initial_points]  # Formatted for evaluator
    params = {i: [] for i in param_names}
    for d in initial_points_dicts:
        for k, v in d.items():
            params[k].append(float(v))

    # Batch out initial_points_dicts and evaluate:
    batches = [initial_points_dicts[i:i + args.batch_size] for i in range(0, len(initial_points_dicts), args.batch_size)]
    all_losses = []
    for b in batches:
        initial_losses, _ = objective.evaluate_batch_without_grads(b)
        all_losses.extend(initial_losses)

    opt_state = optimizer.init(all_losses, params)
    num_samples = args.batch_size  # Num samples selected in parallel is the batch size

    for i in range(args.num_iterations):
        if args.verbose:
            print(f'Beginning iteration {i}')
        key = jax.random.fold_in(loop_key, i)
        new_params = optimizer.sample(key, opt_state, num_samples)
        list_of_params = [
            dict(zip(new_params.keys(), values))
            for values in zip(*new_params.values())
        ]  # This list is already batch_size in shape
        losses, _ = objective.evaluate_batch_without_grads(list_of_params)

        # Fit each pair:
        for pp, ll in zip(list_of_params, losses):
            opt_state = optimizer.fit(opt_state, ll, pp)

        # Note: Look to use something like BoTorch to use the gradient information instead, right now scikit-optimize
        # does not use this information natively
        if args.verbose:
            if isinstance(losses, list):
                print(f"  - Observed losses at {i}: {[f'{l:.4f}' for l in losses]}")
            else:
                print(f"  - Observed loss at {i}: {losses}")

    # Finally, get the best results for returning:
    best_result = opt_state.best_score
    best_params_dict = opt_state.best_params

    if args.verbose:
        print(f"Minimum loss found: {best_result:.6f}")
        print(f"Optimal parameters: {best_params_dict}")

    final_sim_eval_value, final_toml = objective.final_forward_simulation(
        {k: jnp.array(v) for k, v in best_params_dict.items()},
        objective.static_sim_data
    )

    output_toml = final_location / 'optimized_input_file.toml'
    with open(output_toml, "w") as f:
        toml.dump(final_toml, f)

    if args.verbose:
        print(f"Bayesian Optimization finished...")
        print(f"  - Predicted eval at optimum: {final_sim_eval_value.item():.4f}")

    _visualize_gp(opt_state, args, final_location, visualize_gp)
    return best_result, best_params_dict
