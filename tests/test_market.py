import numpy as np
import pandas as pd
import pytest

from src.config import PROJECT_ROOT, load_config
from src.customer_agent import CustomerAgent, ScenarioCustomer
from src.env import HotelPricingEnv
from src.evaluate import _episode_context, _make_policies, _run_one_scenario, evaluate_agents, goal_table
from src.hotels import generate_hotel_population, generate_validation_hotels
from src.market import CompetitorTrack, Market, build_market, generate_competitor_hotels
from src.rl_agent import TabularQLearner
from src.train import train_agent


def _market_config():
    config = load_config()
    config["market"]["enabled"] = True
    return config


def _track(prices, available, sizes, steps=4, hotel_id="c", strategy="fixed_1.00"):
    return CompetitorTrack(
        hotel_id,
        strategy,
        np.asarray(sizes, dtype=float),
        np.tile(np.asarray(prices, dtype=float), (steps, 1)),
        np.tile(np.asarray(available, dtype=bool), (steps, 1)),
    )


def _customer(customer_id, segment, wtp, via_ota=True):
    return ScenarioCustomer(customer_id, 0, segment, float(wtp), via_ota, 0.5, 0.5)


def _signature(decisions):
    return [(d.customer_id, d.status, d.room_index, d.competitor_index) for d in decisions]


def test_disabled_path_reproduces_saved_episode_metrics():
    config = load_config()
    assert config["market"]["enabled"] is False
    q_table = np.load(PROJECT_ROOT / "models" / "q_table_final.npy")
    counts = np.load(PROJECT_ROOT / "models" / "q_counts_final.npy")
    saved = pd.read_csv(PROJECT_ROOT / "results" / "final" / "final_main_episodes.csv")
    _, test = generate_hotel_population(config)
    seed_base = config["evaluation"]["seed_base"]
    policies = _make_policies(q_table, config, seed_base, counts)
    compared = ["net_profit", "occupancy", "booking_count", "cancellation_count", "booked", "too_expensive", "full"]

    for tier_name, hotels in test.items():
        tier_id = config["tiers"][tier_name]["id"]
        for episode_index in range(2):
            seed = seed_base + config["evaluation"]["seed_episode_multiplier"] * episode_index + tier_id
            season, weekend = _episode_context(seed, config)
            rows, _, _, _ = _run_one_scenario(
                hotels[episode_index % len(hotels)], seed, season, weekend, policies, config
            )
            for row in rows:
                expected = saved[
                    saved["policy"].eq(row["policy"])
                    & saved["tier"].eq(tier_name)
                    & saved["episode_index"].eq(episode_index)
                ].iloc[0]
                assert row["competitor"] == 0
                for column in compared:
                    assert row[column] == pytest.approx(expected[column], rel=1e-9), (tier_name, row["policy"], column)


def test_empty_market_matches_no_market():
    config = _market_config()
    _, test = generate_hotel_population(config)
    policies = _make_policies(TabularQLearner(config).q_table, config, 1)[:2]  # Fixed and Rule-based
    seed = 4321
    season, weekend = _episode_context(seed, config)
    hotel = test["midscale"][0]
    plain, _, _, _ = _run_one_scenario(hotel, seed, season, weekend, policies, config)
    empty, _, _, _ = _run_one_scenario(hotel, seed, season, weekend, policies, config, market=Market([]))
    assert plain == empty
    assert Market([]).median_price(0, 0) is None
    assert all(len(array) == 0 for array in Market([]).offers(0))


def test_competitor_pool_and_market_draws_are_deterministic():
    config = _market_config()
    config["market"]["train_competitor_count"] = [2, 3]
    competitors = generate_competitor_hotels(config)
    train, test = generate_hotel_population(config)
    validation = generate_validation_hotels(config)
    for tier in config["tiers"]:
        ids = [hotel.hotel_id for hotel in competitors[tier]]
        assert ids == [f"{tier}_competitor_{index:02d}" for index in range(config["market"]["competitors_per_tier"])]
        other_ids = {hotel.hotel_id for pool in (train, test, validation) for hotel in pool[tier]}
        assert set(ids).isdisjoint(other_ids)
        assert all(hotel.pool == "competitor" for hotel in competitors[tier])
    again = generate_competitor_hotels(config)
    assert [h.to_dict() for h in again["luxury"]] == [h.to_dict() for h in competitors["luxury"]]

    candidates = competitors["budget"]
    excluded = candidates[0].hotel_id
    args = (candidates, (2, 3), "high", True)
    first = build_market(*args, (11,), config, exclude_hotel_id=excluded)
    second = build_market(*args, (11,), config, exclude_hotel_id=excluded)
    other = build_market(*args, (12,), config, exclude_hotel_id=excluded)

    assert 2 <= len(first.competitors) <= 3
    assert len(first.competitors) == len(second.competitors)
    for a, b in zip(first.competitors, second.competitors):
        assert (a.hotel_id, a.strategy) == (b.hotel_id, b.strategy)
        assert np.array_equal(a.prices, b.prices) and np.array_equal(a.available, b.available)
        assert a.strategy in config["market"]["strategies"]
        assert a.hotel_id != excluded
    assert len({t.hotel_id for t in first.competitors}) == len(first.competitors)
    assert [(t.hotel_id, t.strategy) for t in first.competitors] != [
        (t.hotel_id, t.strategy) for t in other.competitors
    ] or not all(
        np.array_equal(a.prices, b.prices) for a, b in zip(first.competitors, other.competitors)
    )
    track = first.competitors[0]
    assert track.prices.shape == track.available.shape == (config["episode"]["steps"], 3)
    assert track.available.dtype == bool


def test_direct_customers_never_choose_competitors():
    config = load_config()
    train, _ = generate_hotel_population(config)
    hotel = train["budget"][0]
    sizes = [room.size_m2 for room in hotel.rooms]
    market = Market([_track([1.0, 1.0, 1.0], [True] * 3, [10 * s for s in sizes])])
    prices = hotel.reference_price_thb * np.array([1.0, 1.4, 2.0])
    rng = np.random.default_rng(3)
    customers = [
        _customer(i, ["leisure", "business"][i % 2], hotel.reference_price_thb * rng.uniform(0.2, 3.0), via_ota=False)
        for i in range(100)
    ]
    agent = CustomerAgent(config)
    rooms_with = np.array([room.count for room in hotel.rooms])
    rooms_without = rooms_with.copy()
    _, with_market = agent.respond(customers, hotel, prices, rooms_with, market.offers(0))
    _, without_market = agent.respond(customers, hotel, prices, rooms_without, None)
    assert _signature(with_market) == _signature(without_market)
    assert all(d.status != "competitor" for d in with_market)
    assert np.array_equal(rooms_with, rooms_without)


def test_ota_customers_choose_max_surplus_room_across_market():
    config = load_config()
    train, _ = generate_hotel_population(config)
    hotel = train["budget"][0]
    agent = CustomerAgent(config)
    reference = hotel.reference_price_thb
    sizes = np.array([room.size_m2 for room in hotel.rooms])
    our_prices = reference * np.array([1.0, 1.4, 2.0])
    tracks = [
        _track(reference * np.array([1.1, 1.5, 2.1]), [True, False, True], sizes * [1.0, 1.1, 1.2]),
        _track(reference * np.array([1.2, 1.45, 2.2]), [True, True, False], sizes * [1.05, 1.0, 1.1]),
    ]
    market = Market(tracks)
    rooms = np.array([3, 0, 2])
    rng = np.random.default_rng(5)
    customers = [
        _customer(i, ["leisure", "business"][i % 2], reference * rng.uniform(0.3, 2.5)) for i in range(300)
    ]

    statuses = set()
    for customer in customers:
        beta = hotel.segments[customer.segment]["size_beta"]
        ours = [
            (customer.wtp_thb * (sizes[r] / sizes.min()) ** beta - our_prices[r], r)
            for r in range(3)
            if rooms[r] > 0
        ]
        theirs = [
            (customer.wtp_thb * (t.sizes_m2[r] / sizes.min()) ** beta - t.prices[0, r], k)
            for k, t in enumerate(tracks)
            for r in range(3)
            if t.available[0, r]
        ]
        our_best = max(ours, key=lambda item: item[0])
        their_best = max(theirs, key=lambda item: item[0])
        if max(our_best[0], their_best[0]) < 0:
            expected = ("too_expensive", None, None)
        elif our_best[0] >= their_best[0]:
            expected = ("booked", our_best[1], None)
        else:
            expected = ("competitor", None, their_best[1])
        bookings, decisions = agent.respond([customer], hotel, our_prices, rooms.copy(), market.offers(0))
        decision = decisions[0]
        assert (decision.status, decision.room_index, decision.competitor_index) == expected
        statuses.add(decision.status)
        assert len(bookings) == int(decision.status == "booked")
    assert statuses == {"booked", "competitor", "too_expensive"}


def test_tie_goes_to_us_and_competitor_choice_keeps_inventory():
    config = load_config()
    train, _ = generate_hotel_population(config)
    hotel = train["budget"][0]
    agent = CustomerAgent(config)
    reference = hotel.reference_price_thb
    sizes = np.array([room.size_m2 for room in hotel.rooms])
    prices = reference * np.array([1.0, 1.4, 2.0])

    # Competitor room identical to our smallest room (size and price); our other rooms are sold out.
    tie = Market([_track([prices[0], 1e9, 1e9], [True, False, False], sizes)])
    rooms = np.array([2, 0, 0])
    _, decisions = agent.respond([_customer(0, "leisure", 2 * reference)], hotel, prices, rooms, tie.offers(0))
    assert (decisions[0].status, decisions[0].room_index) == ("booked", 0)
    assert rooms.tolist() == [1, 0, 0]

    # Cheaper identical competitor room wins, and neither inventory changes.
    cheaper = Market([_track([prices[0] - 100, 1e9, 1e9], [True, False, False], sizes)])
    rooms = np.array([2, 0, 0])
    bookings, decisions = agent.respond(
        [_customer(0, "leisure", 2 * reference)], hotel, prices, rooms, cheaper.offers(0)
    )
    assert (decisions[0].status, decisions[0].room_index, decisions[0].competitor_index) == ("competitor", None, 0)
    assert bookings == [] and rooms.tolist() == [2, 0, 0]

    # We are full and a competitor is acceptable.
    rooms = np.zeros(3, dtype=int)
    _, decisions = agent.respond(
        [_customer(0, "leisure", 2 * reference)], hotel, prices, rooms, cheaper.offers(0)
    )
    assert decisions[0].status == "competitor"

    # Nothing acceptable: "too_expensive" when we have rooms, "full" when we do not.
    dear = Market([_track([1e9, 1e9, 1e9], [True] * 3, sizes)])
    rooms = np.array([2, 2, 2])
    poor = [_customer(0, "leisure", 1.0)]
    assert agent.respond(poor, hotel, prices, rooms, dear.offers(0))[1][0].status == "too_expensive"
    assert agent.respond(poor, hotel, prices, np.zeros(3, dtype=int), dear.offers(0))[1][0].status == "full"


def test_state_bucket_boundaries_use_available_competitors_at_previous_hour():
    config = _market_config()
    train, _ = generate_hotel_population(config)
    hotel = train["budget"][0]
    sizes = [room.size_m2 for room in hotel.rooms]
    cheap = _track([900.0] * 3, [True] * 3, sizes)
    # Competitor B is dear at step 1 but unavailable there; at step 0 both are available.
    dear_prices = np.array([[1100.0] * 3, [5000.0] * 3, [1100.0] * 3, [1100.0] * 3])
    dear_available = np.array([[True] * 3, [False] * 3, [True] * 3, [True] * 3])
    dear = CompetitorTrack("d", "fixed_1.15", np.asarray(sizes), dear_prices, dear_available)
    env = HotelPricingEnv(hotel=hotel, config=config)

    def bucket(our_price, market, step_index):
        env.reset(seed=1, options={"hotel": hotel, "season": "high", "is_weekend": False, "market": market})
        env.step_index = step_index
        env.prices[:] = our_price
        state = env.room_state(0)
        assert len(state) == 8
        return state[-1]

    only_cheap = Market([cheap])  # median 900 at every hour
    assert bucket(900 * 0.95, only_cheap, 2) == 1
    assert bucket(900 * 0.95 - 0.01, only_cheap, 2) == 0
    assert bucket(900 * 1.05, only_cheap, 2) == 1
    assert bucket(900 * 1.05 + 0.01, only_cheap, 2) == 2

    both = Market([cheap, dear])  # median 1000 at hour 0, 900 at hour 1 (B unavailable), 1000 later
    assert bucket(1000.0 * 1.05 + 0.01, both, 1) == 2  # step 1 looks at hour 0
    assert bucket(1000.0 * 1.05, both, 1) == 1
    assert bucket(900.0 * 1.05 + 0.01, both, 2) == 2  # step 2 looks at hour 1
    assert bucket(900.0 * 1.05, both, 2) == 1
    assert bucket(1000.0 * 1.05, both, 0) == 1  # step 0 uses hour 0

    nobody = Market([_track([900.0] * 3, [False] * 3, sizes)])
    assert bucket(1.0, nobody, 2) == 1
    assert bucket(1.0, None, 2) == 1


def test_q_table_shape_depends_on_market_flag():
    off = load_config()
    q_off = TabularQLearner(off).q_table
    assert int(np.prod(q_off.shape[:-1])) == 19_440 and q_off.shape[-1] == 7
    train, _ = generate_hotel_population(off)
    env = HotelPricingEnv(hotel=train["budget"][0], config=off)
    env.reset(seed=1, options={"hotel": train["budget"][0]})
    assert len(env.room_state(0)) == 7

    on = _market_config()
    q_on = TabularQLearner(on).q_table
    assert int(np.prod(q_on.shape[:-1])) == 58_320 and q_on.shape[-1] == 7
    env = HotelPricingEnv(hotel=train["budget"][0], config=on)
    env.reset(seed=1, options={"hotel": train["budget"][0]})
    assert len(env.room_state(0)) == 8


def test_train_and_evaluate_smoke_with_market(tmp_path):
    config = _market_config()
    config["paths"]["models"] = str(tmp_path / "models")
    config["paths"]["results"] = str(tmp_path / "results")
    config["rl"]["checkpoint_interval_episodes"] = 0
    config["market"]["train_competitor_count"] = [2, 2]
    config["market"]["eval_competitor_count"] = 2

    learner, _, _ = train_agent(config, episodes_per_tier=1, tiers=["budget"])
    assert learner.q_table.shape[:-1] == (3, 3, 3, 2, 10, 9, 4, 3)
    assert learner.update_counts.sum() > 0

    episodes, _ = evaluate_agents(
        learner.q_table, config, episodes_per_tier=1, output_stem="market_smoke", update_counts=learner.update_counts
    )
    goals = pd.read_csv(tmp_path / "results" / "market_smoke_goals.csv")
    for column in ["conversion", "ota_market_share", "competitor", "meets_floor", "net_profit_pct_vs_fixed"]:
        assert column in goals.columns
    assert (episodes["competitor"] >= 0).all()
    assert goal_table(episodes, ["model", "tier"], 0.03).shape[0] == goals.shape[0]
