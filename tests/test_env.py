import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.baselines import FixedPolicy, RuleBasedPolicy
from src.config import load_config
from src.env import HotelPricingEnv
from src.hotels import generate_hotel_population


@pytest.fixture(scope="module")
def project_setup():
    config = load_config()
    train, _ = generate_hotel_population(config)
    return config, train


@pytest.mark.parametrize("tier", ["budget", "midscale", "luxury"])
@pytest.mark.parametrize("policy_type", [FixedPolicy, RuleBasedPolicy])
def test_full_episode_respects_masks_inventory_and_reward(project_setup, tier, policy_type):
    config, train = project_setup
    hotel = train[tier][0]
    env = HotelPricingEnv(hotel=hotel, config=config)
    policy = policy_type(config)
    observation, _ = env.reset(
        seed=8123,
        options={
            "hotel": hotel,
            "season": "high",
            "is_weekend": True,
            "scenario_seed": 8123,
        },
    )
    assert env.observation_space.contains(observation)
    total_reward = 0.0
    previous_levels = env.levels.copy()
    steps = 0

    while True:
        actions = policy.select_actions(env)
        observation, reward, terminated, truncated, info = env.step(actions)
        total_reward += reward
        steps += 1

        assert not truncated
        assert env.observation_space.contains(observation)
        assert np.all(env.rooms_left >= 0)
        assert all(action in mask for action, mask in zip(actions, info["action_masks"]))
        for room_index in range(1, len(info["prices"])):
            assert info["prices"][room_index] >= (
                config["pricing"]["minimum_large_room_price_ratio"]
                * info["prices"][room_index - 1]
            )
        if steps > 1:
            for room_index, forced in enumerate(info["forced_large_room_prices"]):
                if not forced:
                    assert abs(info["action_levels"][room_index] - previous_levels[room_index]) <= config[
                        "pricing"
                    ]["max_level_change_per_hour"]
        previous_levels = np.asarray(info["action_levels"])

        if terminated:
            assert steps == config["episode"]["steps"]
            assert total_reward == pytest.approx(info["episode_metrics"]["net_profit"])
            assert total_reward == pytest.approx(info["episode_metrics"]["reward_sum"])
            break


def test_gymnasium_contract_and_common_random_scenario(project_setup):
    config, train = project_setup
    hotel = train["midscale"][0]
    env = HotelPricingEnv(hotel=hotel, config=config)
    check_env(env, skip_render_check=True)
    first = env.customer_agent.generate_scenario(hotel, "shoulder", False, 9876)
    second = env.customer_agent.generate_scenario(hotel, "shoulder", False, 9876)
    assert first == second
    assert len(first.customers_by_step) == config["episode"]["steps"]
