from src.config import load_config


def test_config_has_three_tiers_and_720_hour_episode():
    config = load_config()
    assert config["episode"]["steps"] == 720
    assert list(config["tiers"]) == ["budget", "midscale", "luxury"]
