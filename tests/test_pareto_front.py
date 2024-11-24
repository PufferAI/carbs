import random
import json

import numpy as np
import torch

from carbs import LinearSpace
from carbs import LogSpace
from carbs import LogitSpace
from carbs import Param
from carbs import CARBS
from carbs import CARBSParams
from carbs import ObservationInParam


RANDOM_SEED = 42
NUM_SUGGESTIONS = 30
NUM_RANDOM_SAMPLES = 7
RESAMPLE_FREQUENCY = 5

TEST_ARGS = {
    "train": {
        "total_timesteps": 1_000_000,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "update_epochs": 4,
    },
    "sweep": {
        "metric": {
            "goal": "maximize",
            "name": "environment/episode_return",
        },
        "parameters": {
            "train": {
                "parameters": {
                    "total_timesteps": {"min": 1_000_000, "max": 10_000_000},
                    "learning_rate": {"min": 1e-5, "max": 1e-1},
                    "gamma": {"min": 0.0, "max": 1.0},
                    "gae_lambda": {"min": 0.0, "max": 1.0},
                    "update_epochs": {"min": 1, "max": 20},
                }
            }
        }
    },
    "carbs": {
        "total_timesteps": { "min": 1_000_000, "space": "log", "search_center": 1_000_000 },
        "learning_rate": { "space": "log", "search_center": 3e-4 },
        "gamma": { "space": "logit", "search_center": 0.97 },
        "gae_lambda": { "space": "logit", "search_center": 0.90 },
        "update_epochs": { "space": "linear", "search_center": 4 },
    }
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    if seed is not None:
        torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def carbs_param(group, name, carbs_kwargs, rounding_factor=1):
    # carbs_kwargs should have either ("min", "max") or "values"
    if "values" in carbs_kwargs:
        values = carbs_kwargs["values"]
        mmin = min(values)
        mmax = max(values)
    else:
        mmin = carbs_kwargs["min"]
        mmax = carbs_kwargs["max"] if "max" in carbs_kwargs else float("+inf")

    space = carbs_kwargs["space"]
    search_center = carbs_kwargs["search_center"] if "search_center" in carbs_kwargs else None
    is_integer = carbs_kwargs["is_integer"] if "is_integer" in carbs_kwargs else False

    if space == "log":
        Space = LogSpace
    elif space == "linear":
        Space = LinearSpace
    elif space == "logit":
        Space = LogitSpace
        assert mmin == 0
        assert mmax == 1
        assert search_center is not None
    else:
        raise ValueError(f"Invalid CARBS space: {space} (log/linear)")

    return Param(
        name=f"{group}-{name}",
        space=Space(min=mmin, max=mmax, is_integer=is_integer, rounding_factor=rounding_factor),
        search_center=search_center,
    )


def init_carbs(args, resample_frequency=5, num_random_samples=5, max_suggestion_cost=600):
    assert "sweep" in args, "No wandb sweep config found in args"
    assert "carbs" in args, "No carbs config found in args"

    carbs_param_spaces = []
    wandb_sweep_params = args["sweep"]["parameters"]
    carbs_config = args["carbs"]

    for group in wandb_sweep_params:
        for name in wandb_sweep_params[group]["parameters"]:
            assert name in carbs_config, f"Invalid name {name} in {group}"

            # Handle special cases: total timesteps, batch size, num_minibatch
            if name in ["total_timesteps", "batch_size", "num_minibatches", "bptt_horizon"]:
                assert (
                    "min" in carbs_config[name]
                ), f"Special param {name} must have min in carbs config"

            # Others: append min/max from wandb param to carbs param
            else:
                carbs_config[name].update(wandb_sweep_params[group]["parameters"][name])

            carbs_param_spaces.append(
                carbs_param(group, name, carbs_config[name], rounding_factor=1)
            )

    carbs_params = CARBSParams(
        better_direction_sign=1,
        is_wandb_logging_enabled=False,
        resample_frequency=resample_frequency,
        num_random_samples=num_random_samples,
        max_suggestion_cost=max_suggestion_cost,
    )

    return CARBS(carbs_params, carbs_param_spaces)


def print_pareto_front(carbs, is_conservative=False):
    pareto_front = carbs._get_pareto_groups(is_conservative=is_conservative)
    print(f"\nPareto front (conservative = {is_conservative}):")
    for i, obs_group in enumerate(pareto_front):
        mean_cost = np.mean([o.cost for o in obs_group])
        mean_output = np.mean([o.output for o in obs_group])
        print(
            f"Obs group {i+1}, {len(obs_group)} samples - cost: {mean_cost:.2f}, mean output: {mean_output:.2f}"
        )


def carbs_runner_fn(args, env_name, carbs, sweep_id, train_fn):
    target_metric = args["sweep"]["metric"]["name"].split("/")[-1]
    carbs_file = "carbs_" + sweep_id + ".txt"

    def run_sweep_session():
        print("--------------------------------------------------------------------------------")
        print("Starting a new session...")
        print("--------------------------------------------------------------------------------")

        print(f"Getting suggestion based on CARBS's {len(carbs.success_observations)} obs...")

        orig_suggestion = carbs.suggest().suggestion
        suggestion = orig_suggestion.copy()
        print("\nCARBS suggestion:", suggestion)
        train_suggestion = {
            k.split("-")[1]: v for k, v in suggestion.items() if k.startswith("train-")
        }

        # Correcting critical parameters before updating
        # train_suggestion["total_timesteps"] = int(train_suggestion["total_timesteps"] * 10**6)
        # for key in ["batch_size", "bptt_horizon"]:
        #     if key in train_suggestion:
        #         train_suggestion[key] = 2 ** round(train_suggestion[key])
        train_suggestion["update_epochs"] = round(train_suggestion["update_epochs"])

        # CARBS minibatch_size is actually the number of minibatches
        # train_suggestion["num_minibatches"] = 2 ** round(train_suggestion["num_minibatches"])
        # train_suggestion["minibatch_size"] = (
        #     train_suggestion["batch_size"] // train_suggestion["num_minibatches"]
        # )

        # args["train"]["num_envs"] = closest_power(train_suggestion["num_envs"])  # 16, 32, 64
        args["train"].update(train_suggestion)

        stats, uptime, is_success = {}, 0, False
        try:
            stats, uptime = train_fn(args)
            is_success = len(stats) > 0
        except Exception as e:  # noqa
            import traceback

            traceback.print_exc()

        # NOTE: What happens if training fails?
        """
        A run should be reported as a failure if the hyperparameters suggested by CARBS 
        caused the failure, for example a batch size that is too large that caused an OOM failure. 
        If a failure occurs that is not related to the hyperparameters, it is better to forget 
        the suggestion or retry it. Report a failure by making an ObservationInParam with is_failure=True
        """
        observed_value = [s[target_metric] for s in stats if target_metric in s]
        if len(observed_value) > 0:
            observed_value = np.mean(observed_value)
        else:
            observed_value = 0

        print(f"\nTrain success: {is_success}, Observed value: {observed_value}\n")
        obs_out = carbs.observe(  # noqa
            ObservationInParam(
                input=orig_suggestion,
                output=observed_value,
                cost=uptime,
                is_failure=not is_success,
            )
        )

        # Save CARBS suggestions and results
        with open(carbs_file, "a") as f:
            train_suggestion.update({"output": observed_value, "cost": uptime})
            results_txt = json.dumps(train_suggestion)
            f.write(results_txt + "\n")
            f.flush()

    return run_sweep_session



if __name__ == "__main__":
    carbs = init_carbs(TEST_ARGS, resample_frequency=RESAMPLE_FREQUENCY, num_random_samples=NUM_RANDOM_SAMPLES)

    rng = np.random.default_rng(RANDOM_SEED)

    def dummy_train_fn(args):
        seed_everything(RANDOM_SEED)

        stats = [{"episode_return": rng.integers(1, 100)}]
        cost = args["train"]["total_timesteps"] // 10000

        return stats, cost

    carbs_runner = carbs_runner_fn(
        TEST_ARGS, "test", carbs, "test", train_fn=dummy_train_fn
    )

    for i in range(NUM_SUGGESTIONS):
        carbs_runner()
        # print("CARBS state:", carbs.get_state_dict(), "\n")
        print_pareto_front(carbs)
        print_pareto_front(carbs, is_conservative=True)
