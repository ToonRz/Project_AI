import numpy as np

from src.config import load_config
from src.env import HotelPricingEnv
from src.hotels import generate_hotel_population
from src.rl_agent import QTablePolicy, TabularQLearner


def test_n_step_bootstrap_uses_only_the_next_state_mask():
    config = load_config()
    learner = TabularQLearner(config, algorithm="n_step")
    state = (0, 0, 0, 0, 0, 8, 0)
    next_state = (0, 0, 0, 0, 0, 7, 0)
    learner.begin_episode()
    for _ in range(config["rl"]["n_step"]):
        learner.record(0, state, 0, 1.0)
    learner.q_table[next_state + (2,)] = 10.0
    learner.q_table[next_state + (3,)] = 100.0
    learner.q_table[next_state + (6,)] = 1000.0

    learner.update_ready(0, next_state, 2, 3)

    expected_target = config["rl"]["n_step"] + 100.0
    assert learner.q_table[state + (0,)] == expected_target


def test_terminal_flush_does_not_bootstrap():
    config = load_config()
    learner = TabularQLearner(config, algorithm="n_step")
    first_state = (0, 0, 0, 0, 0, 8, 0)
    second_state = (0, 0, 0, 0, 1, 7, 0)
    learner.begin_episode()
    learner.record(0, first_state, 1, 2.0)
    learner.record(0, second_state, 2, 1.0)
    learner.q_table[first_state + (6,)] = 10000.0

    learner.finish_episode()

    assert learner.q_table[first_state + (1,)] == 3.0
    assert learner.q_table[second_state + (2,)] == 1.0


def test_monte_carlo_control_uses_full_terminal_return():
    config = load_config()
    learner = TabularQLearner(config, algorithm="monte_carlo")
    first_state = (0, 0, 0, 0, 0, 8, 0)
    second_state = (0, 0, 0, 0, 1, 7, 0)
    learner.begin_episode()
    learner.record(0, first_state, 1, 2.0)
    learner.record(0, second_state, 2, 1.0)

    learner.finish_episode()

    assert learner.q_table[first_state + (1,)] == 3.0
    assert learner.q_table[second_state + (2,)] == 1.0


def test_epsilon_greedy_actions_stay_inside_mask():
    config = load_config()
    learner = TabularQLearner(config, algorithm="n_step")
    state = (0, 0, 0, 0, 0, 8, 0)
    rng = np.random.default_rng(515)
    sampled = {learner.choose_action(state, 3, 4, 1.0, rng) for _ in range(100)}
    assert sampled == {3, 4}


def test_delayed_cancellation_reward_is_attached_to_prior_room_transition():
    config = load_config()
    learner = TabularQLearner(config, algorithm="n_step")
    previous_state = (0, 0, 0, 0, 0, 8, 0)
    current_state = (0, 0, 0, 0, 1, 7, 0)
    learner.begin_episode()
    learner.record(0, previous_state, 2, 100.0)
    learner.add_reward_to_last_transition(0, -30.0)
    learner.record(0, current_state, 3, 0.0)

    learner.finish_episode()

    assert learner.q_table[previous_state + (2,)] == 70.0
    assert learner.q_table[current_state + (3,)] == 0.0


def test_unseen_state_uses_reference_markup_clamped_to_mask():
    config = load_config()
    train_hotels, _ = generate_hotel_population(
        config, mode=config["synthetic_hotels"]["single_profile_mode"]
    )
    hotel = train_hotels["budget"][0]
    env = HotelPricingEnv(hotel=hotel, config=config)
    env.reset(seed=10, options={"hotel": hotel, "season": "shoulder", "is_weekend": False})
    learner = TabularQLearner(config)
    policy = QTablePolicy(learner, config, seed=10)

    actions = policy.select_actions(env)
    target = config["rl"]["unseen_state_markup_level"]
    assert actions.tolist() == [target, target, target]

    env.reset(seed=11, options={"hotel": hotel, "season": "shoulder", "is_weekend": False})
    env.step_index = 1
    env.hours_remaining = config["episode"]["steps"] - env.step_index
    env.levels[:] = 0
    env.prices = np.asarray(
        [env.price_for_level(index, 0) for index in range(len(hotel.rooms))]
    )
    env.rooms_left = np.asarray([room.count for room in hotel.rooms], dtype=np.int64)
    clamped_actions = policy.select_actions(env)
    assert clamped_actions.tolist() == [1, 1, 1]

    state = env.room_state(0)
    learner.q_table[state + (0,)] = 100.0
    learner.update_counts[state + (0,)] = 1
    env.reset(seed=10, options={"hotel": hotel, "season": "shoulder", "is_weekend": False})
    assert policy.select_actions(env)[0] == 0


def test_epsilon_start_scales_schedule_and_keeps_default():
    config = load_config()
    learner = TabularQLearner(config)
    assert learner.epsilon(0, 100) == 1.0
    assert learner.epsilon(0, 100, start=0.3) == 0.3
    assert learner.epsilon(99, 100, start=0.3) == config["rl"]["epsilon_min"]


def test_warm_start_trains_only_requested_tier_from_behavior_actions(tmp_path):
    from src.baselines import FixedPolicy
    from src.train import train_agent

    config = load_config()
    config["paths"]["models"] = str(tmp_path / "models")
    config["paths"]["results"] = str(tmp_path / "results")
    config["rl"]["checkpoint_interval_episodes"] = 0
    learner = TabularQLearner(config)
    learner.q_table[1:] = 7.0

    trained, _, _ = train_agent(
        config,
        episodes_per_tier=1,
        learner=learner,
        tiers=["budget"],
        behavior_policy=FixedPolicy(config),
    )

    assert trained is learner
    assert np.all(trained.q_table[1:] == 7.0)
    assert trained.update_counts[1:].sum() == 0
    fixed_level = config["baselines"]["fixed_markup_level"]
    assert trained.update_counts[0].sum() > 0
    assert trained.update_counts[0][..., fixed_level].sum() == trained.update_counts[0].sum()


def test_occupancy_penalties_cover_only_the_shortfall():
    from src.train import occupancy_penalties

    assert occupancy_penalties(np.array([2, 1, 0]), 20, 0.80, 100.0).sum() == 0.0
    penalties = occupancy_penalties(np.array([6, 2, 0]), 20, 0.80, 100.0)
    assert np.isclose(penalties.sum(), 100.0 * (0.80 * 20 - 12))
    assert np.allclose(penalties, [300.0, 100.0, 0.0])


def test_occupancy_penalty_only_changes_end_of_episode_values(tmp_path):
    from src.train import train_agent

    def run(penalty_per_room):
        config = load_config()
        config["paths"]["models"] = str(tmp_path / f"models_{penalty_per_room}")
        config["paths"]["results"] = str(tmp_path / f"results_{penalty_per_room}")
        config["rl"]["checkpoint_interval_episodes"] = 0
        config["rl"]["occupancy_penalty_per_room"] = penalty_per_room
        config["rl"]["occupancy_floor_margin"] = -1.0
        learner, curve, _ = train_agent(config, episodes_per_tier=1, tiers=["luxury"])
        return learner.q_table, curve

    plain, plain_curve = run(0.0)
    penalized, penalized_curve = run(1.0)

    assert plain_curve["occupancy_penalty"].iloc[0] == 0.0
    assert penalized_curve["occupancy_penalty"].iloc[0] > 0.0
    assert plain_curve["net_profit"].iloc[0] == penalized_curve["net_profit"].iloc[0]
    changed = np.argwhere(~np.isclose(plain, penalized))
    last_day_bucket = 0
    assert len(changed) > 0
    assert set(changed[:, 4]) == {last_day_bucket}
