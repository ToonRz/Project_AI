import json

from src.config import PROJECT_ROOT, load_config
from src.hotels import generate_hotel_population, generate_validation_hotels


def _rounded(value):
    """Ignore last-bit float differences between machines when comparing saved hotels."""
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    if isinstance(value, float):
        return round(value, 9)
    return value


def test_validation_pool_is_disjoint_and_leaves_train_test_unchanged():
    config = load_config()
    train, test = generate_hotel_population(config)
    validation = generate_validation_hotels(config)

    saved = json.loads((PROJECT_ROOT / "results" / "synthetic_hotels.json").read_text(encoding="utf-8"))
    for pool, hotels_by_tier in [("train", train), ("test", test)]:
        generated = {tier: [hotel.to_dict() for hotel in hotels] for tier, hotels in hotels_by_tier.items()}
        assert _rounded(json.loads(json.dumps(generated))) == _rounded(saved[pool])

    for tier in config["tiers"]:
        assert len(validation[tier]) == config["synthetic_hotels"]["validation_per_tier"]
        assert all(hotel.pool == "validation" and hotel.tier == tier for hotel in validation[tier])
        validation_ids = {hotel.hotel_id for hotel in validation[tier]}
        other_ids = {hotel.hotel_id for hotel in train[tier] + test[tier]}
        assert len(validation_ids) == len(validation[tier])
        assert validation_ids.isdisjoint(other_ids)
        test_params = {(hotel.reference_price_thb, hotel.demand_per_room_episode) for hotel in test[tier]}
        assert all(
            (hotel.reference_price_thb, hotel.demand_per_room_episode) not in test_params
            for hotel in validation[tier]
        )

    again = generate_validation_hotels(config)
    assert [hotel.to_dict() for hotel in again["midscale"]] == [hotel.to_dict() for hotel in validation["midscale"]]
