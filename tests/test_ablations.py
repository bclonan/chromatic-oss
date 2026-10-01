from experiments.ablations import run_robustness


def test_payload_corruption_rejects_invalid_palette_colors():
    records = run_robustness({"seeds": [7]})
    chroma = next(r for r in records if r["representation"] == "chroma8" and r["probe"] == "payload LSB flip")
    integer = next(r for r in records if r["representation"] == "int8" and r["probe"] == "payload LSB flip")
    assert chroma["invalid_symbol_fraction"] > 0.9
    assert integer["invalid_symbol_fraction"] == 0
    depths = [r for r in records if r["probe"] == "precision depth"]
    assert depths[0]["tokens_per_value"] < depths[-1]["tokens_per_value"]
    assert depths[0]["mae"] > depths[-1]["mae"]
