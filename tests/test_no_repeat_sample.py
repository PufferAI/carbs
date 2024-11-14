from carbs import LinearSpace
from carbs import LogSpace
from carbs import LogitSpace
from carbs import Param
from carbs import CARBS
from carbs import CARBSParams
from carbs import ObservationInParam


def test_no_repeat_random_suggestions(carbs_instance: CARBS) -> None:
    num_to_suggest = 30
    for i in range(num_to_suggest):
        # Simulate a case where the same seed is kept set elsewhere
        carbs_instance._set_seed(1)
        suggestion = carbs_instance._get_random_suggestion()
        observation = ObservationInParam(input=suggestion.suggestion, output=i, cost=i + 1)
        carbs_instance.observe(observation)
        print(suggestion.suggestion)

if __name__ == "__main__":

    param_spaces = [
        Param(
            name="total_timesteps",
            space=LinearSpace(min=10_000_000, max=50_000_000, scale=10_000_000, is_integer=True),
            search_center=20_000_000,
        ),
        Param(name="learning_rate", space=LogSpace(min=1e-5, max=1e-1), search_center=3e-05),
        Param(name="gamma", space=LogitSpace(min=0.8, max=0.9999), search_center=0.995),
        Param(name="gae_lambda", space=LogitSpace(min=0.8, max=1.0), search_center=0.98),

        # NOTE: after casting to integer, the suggestion is no longer "close" in basic space.
        # So, it escapes torch.isclose() is false and keeps getting resampled.
        # Param(
        #     name="update_epochs", space=LinearSpace(min=1, max=15, scale=5, is_integer=True), search_center=5
        # ),
        Param(
            name="update_epochs", space=LinearSpace(min=1, max=15, scale=5), search_center=5
        ),
    ]

    carbs_params = CARBSParams(
        better_direction_sign=1,
        is_wandb_logging_enabled=False, 
        resample_frequency=5,
        num_random_samples=5,
    )

    carbs = CARBS(carbs_params, param_spaces)

    test_no_repeat_random_suggestions(carbs)
