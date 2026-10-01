"""Exercise the real cube backend and its HTTP request boundaries."""
import copy
import json
import math
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from dashboard.cube_api import inspect_cube
from dashboard.server import DashboardServer


def create(**kwargs):
    return inspect_cube({"dimensions": 4, "size": 3, "seed": 31,
                         "holographic": False, **kwargs})


def operate(state, operation="view", **kwargs):
    return inspect_cube({"state": state, "operation": operation,
                         "holographic": False, **kwargs})


def cells_by_id(result):
    return {cell["id"]: cell for cell in result["all_cells"]}


def test_create_has_every_real_cell_separate_payload_and_identity_counts():
    result = create(dimensions=6)
    assert result["state"]["shape"] == [3] * 6
    assert len(result["all_cells"]) == 729
    assert result["metrics"]["total_cells"] == 729
    assert result["metrics"]["visible_cells"] == 27
    assert result["metrics"]["payload_bytes"] == 729 * 3
    assert result["metrics"]["identity_bytes"] == 729 * 8
    assert result["metrics"]["moved_cells"] == 0
    assert len(result["metrics"]["state_sha256"]) == 64
    assert "not trained model data" in result["source"]
    for index, cell in enumerate(result["all_cells"]):
        assert cell["id"] == index
        assert cell["codes"] == result["state"]["codes"][index]
        assert cell["rgb"] == cell["codes"]
        assert len(cell["full_coordinate"]) == 6


def test_integer_twists_move_cell_identity_and_value_then_inverse_restores_state():
    original = create(dimensions=6, signal="noise")
    moved = operate(original["state"], "twist", axes=[5, 2], layer_axis=0,
                    layer_index=1, turns=1)
    assert moved["metrics"]["moved_cells"] > 0
    assert moved["metrics"]["twist_count"] == 1
    assert moved["metrics"]["state_sha256"] != original["metrics"]["state_sha256"]
    original_cells, moved_cells = cells_by_id(original), cells_by_id(moved)
    assert moved_cells.keys() == original_cells.keys()
    assert any(moved_cells[index]["full_coordinate"] != original_cells[index]["full_coordinate"]
               for index in moved_cells)
    for index in moved_cells:
        assert moved_cells[index]["codes"] == original_cells[index]["codes"]
        assert moved_cells[index]["rgb"] == original_cells[index]["rgb"]
        assert moved_cells[index]["value"] == original_cells[index]["value"]
    restored = operate(moved["state"], "twist", axes=[5, 2], layer_axis=0,
                       layer_index=1, turns=-1)
    assert restored["state"]["codes"] == original["state"]["codes"]
    assert restored["state"]["cell_ids"] == original["state"]["cell_ids"]
    assert restored["metrics"]["state_sha256"] == original["metrics"]["state_sha256"]
    assert restored["metrics"]["moved_cells"] == 0


def test_four_layer_turns_and_reset_recover_seeded_initial_cube():
    original = create(signal="noise")
    turned = original
    for _ in range(4):
        turned = operate(turned["state"], "twist", axes=[0, 1], layer_axis=3, layer_index=2)
    assert turned["metrics"]["state_sha256"] == original["metrics"]["state_sha256"]
    assert turned["metrics"]["twist_count"] == 4
    reset = create(operation="reset", signal="noise")
    assert reset["state"] == original["state"]
    assert reset["all_cells"] == original["all_cells"]


def test_display_axes_order_and_hidden_coordinates_select_real_six_dimensional_slice():
    original = create(dimensions=6)
    result = operate(original["state"], view_axes=[5, 2, 0], fixed={"1": 2, "3": 1, "4": 2})
    assert result["view"]["axes"] == [5, 2, 0]
    assert result["view"]["axis_names"] == ["axis_5", "axis_2", "axis_0"]
    assert result["view"]["shape"] == [3, 3, 3]
    assert result["view"]["fixed"] == {"1": 2, "3": 1, "4": 2}
    assert len(result["view"]["cells"]) == 27
    assert result["metrics"]["state_sha256"] == original["metrics"]["state_sha256"]
    original_cells = cells_by_id(original)
    for cell in result["view"]["cells"]:
        coordinate = cell["full_coordinate"]
        assert coordinate[1] == 2 and coordinate[3] == 1 and coordinate[4] == 2
        assert cell["coordinate"] == [coordinate[5], coordinate[2], coordinate[0]]
        assert cell["value"] == original_cells[cell["id"]]["value"]
    other = operate(original["state"], view_axes=[5, 2, 0], fixed=[0, 0, 0, 0, 0, 0])
    assert {cell["id"] for cell in other["view"]["cells"]}.isdisjoint(
        {cell["id"] for cell in result["view"]["cells"]})


def test_recorded_tensor_source_preserves_values_and_provenance(tmp_path):
    path = tmp_path / "latest.json"
    values = [math.sin(index / 7) for index in range(100)]
    path.write_text(json.dumps({"tensor_samples": [{"name": "Recorded example", "values": values}]}))
    result = inspect_cube({"dimensions": 4, "size": 3, "source_key": "Recorded example",
                           "holographic": False}, path)
    assert "Recorded example" in result["source"]
    assert len(result["all_cells"]) == 81
    assert max(abs(cell["value"] - expected) for cell, expected in zip(result["all_cells"], values)) < 2e-7
    with pytest.raises(ValueError, match="too few"):
        inspect_cube({"dimensions": 5, "size": 3, "source_key": "Recorded example",
                      "holographic": False}, path)


def test_holographic_noise_reconstruction_uses_actual_retained_coefficients_and_errors():
    original = create(signal="noise")
    measurements = {}
    for coverage in (0, 0.5, 1):
        result = operate(original["state"], holographic=True, coverage=coverage)
        hologram = result["holographic"]
        assert len(hologram["cells"]) == len(hologram["decoded_cells"]) == 81
        kept = sum(cell["kept"] for cell in hologram["cells"])
        assert hologram["metrics"]["retained_coefficients"] == kept
        assert hologram["metrics"]["total_coefficients"] == 81
        assert hologram["coverage"] == pytest.approx(kept / 81)
        assert hologram["metrics"]["payload_bytes"] == 81 * 3
        errors = [abs(recovered["value"] - source["value"])
                  for recovered, source in zip(hologram["decoded_cells"], result["all_cells"])]
        assert hologram["metrics"]["mae"] == pytest.approx(sum(errors) / 81)
        assert hologram["metrics"]["mse"] == pytest.approx(sum(error ** 2 for error in errors) / 81)
        assert hologram["metrics"]["max_error"] == pytest.approx(max(errors))
        for cell, error in zip(hologram["decoded_cells"], errors):
            assert cell["error"] == pytest.approx(error)
        measurements[coverage] = hologram["metrics"]
        if coverage == 0:
            assert kept == 0
            assert all(cell["value"] == 0 for cell in hologram["decoded_cells"])
        elif coverage == 1:
            assert kept == 81
            assert hologram["metrics"]["mae"] < 0.02
        else:
            assert kept in (40, 41)
    assert measurements[1]["mse"] < measurements[0.5]["mse"] < measurements[0]["mse"]


@pytest.mark.parametrize("payload", [
    {"dimensions": 2}, {"dimensions": 7}, {"dimensions": True},
    {"dimensions": 6, "size": 8}, {"size": 9}, {"seed": -1},
    {"signal": "unknown"}, {"operation": "delete"}, {"holographic": "false"},
    {"view_axes": [0, 1, 1]}, {"view_axes": [0, 1, 6]},
    {"fixed": {"3.5": 0}}, {"fixed": {3.5: 0}}, {"fixed": {True: 0}},
    {"fixed": {"3": 0, "03": 1}}, {"fixed": {"3": 3}},
])
def test_cube_rejects_invalid_or_unbounded_creation(payload):
    with pytest.raises(ValueError):
        create(**payload)


@pytest.mark.parametrize("field,value", [
    ("cell_ids", [0] * 81), ("codes", [[0, 0, 256]] * 81),
    ("codes", [[0, 0, False]] * 81), ("codes", [[0, 0]] * 81),
    ("axis_names", ["same"] * 4), ("shape", [8] * 6),
    ("twist_count", -1), ("initial_seed", True),
    ("codec", {"name": "rgbd"}),
    ("codec", {"name": "rgb", "minimum": -1e100, "maximum": 1e100, "logarithmic": False}),
    ("codec", {"name": "rgb", "minimum": -1, "maximum": 1, "logarithmic": "false"}),
])
def test_cube_rejects_invalid_imported_state(field, value):
    state = copy.deepcopy(create()["state"])
    state[field] = value
    with pytest.raises(ValueError):
        operate(state)


def test_cube_rejects_invalid_layer_extent_and_twist_count_without_server_failure():
    state = create()["state"]
    with pytest.raises(ValueError, match="layer index"):
        operate(state, "twist", layer_index=3)
    state["twist_count"] = 10000
    with pytest.raises(ValueError, match="twist count"):
        operate(state, "twist")


@pytest.mark.parametrize("coverage", [-0.1, 1.1, True, "0.5", float("nan"), float("inf")])
def test_cube_rejects_invalid_holographic_coverage(coverage):
    with pytest.raises(ValueError, match="coverage"):
        create(holographic=True, coverage=coverage)


@pytest.fixture
def cube_http(tmp_path):
    path = tmp_path / "latest.json"
    path.write_text(json.dumps({"schema_version": 1, "experiments": []}))
    (tmp_path / "cube.json").write_text(json.dumps({"schema_version": 1,
        "geometry": [{"dimensions": 6, "four_turns_exact": True}],
        "reconstruction": [], "experiments": [{"id": "cube-http-fixture"}]}))
    server = DashboardServer(("127.0.0.1", 0), path)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    worker.join(timeout=3)


def post(base, payload, headers=None):
    request = Request(base + "/api/cube", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json", **(headers or {})})
    return urlopen(request)


def test_http_cube_endpoint_runs_real_geometry_and_inverse(cube_http):
    payload = {"dimensions": 4, "size": 3, "signal": "noise", "seed": 7,
               "holographic": False}
    with post(cube_http, payload) as response:
        original = json.load(response)
        assert original == inspect_cube(payload)
        assert response.headers["Cache-Control"] == "no-store"
    turn = {"operation": "twist", "state": original["state"], "axes": [0, 3],
            "layer_axis": 1, "layer_index": 2, "holographic": False}
    with post(cube_http, turn) as response:
        moved = json.load(response)
        assert moved["metrics"]["moved_cells"] > 0
    with post(cube_http, {**turn, "state": moved["state"], "turns": -1}) as response:
        restored = json.load(response)
        assert restored["metrics"]["state_sha256"] == original["metrics"]["state_sha256"]


def test_http_cube_boundaries_reject_cross_origin_bad_state_and_large_body(cube_http):
    with pytest.raises(HTTPError) as error:
        post(cube_http, {"holographic": False}, {"Origin": "https://example.com"})
    assert error.value.code == 403
    state = create()["state"]
    state["cell_ids"][1] = state["cell_ids"][0]
    with pytest.raises(HTTPError) as error:
        post(cube_http, {"operation": "view", "state": state, "holographic": False})
    assert error.value.code == 400
    assert "identities" in json.load(error.value)["error"]
    request = Request(cube_http + "/api/cube", data=b"{}", method="POST",
                      headers={"Content-Type": "application/json", "Content-Length": str(1024 * 1024 + 1)})
    with pytest.raises(HTTPError) as error:
        urlopen(request)
    assert error.value.code == 413


def test_http_cube_returns_bad_request_for_invalid_layer_and_content_type(cube_http):
    with pytest.raises(HTTPError) as error:
        post(cube_http, {"operation": "twist", "state": create()["state"],
                        "layer_index": 3, "holographic": False})
    assert error.value.code == 400
    with pytest.raises(HTTPError) as error:
        post(cube_http, {}, {"Content-Type": "text/plain"})
    assert error.value.code == 415


def test_http_cube_results_reads_its_own_record_beside_latest(cube_http):
    with urlopen(cube_http + "/api/cube-results") as response:
        result = json.load(response)
        assert result["schema_version"] == 1
        assert result["experiments"] == [{"id": "cube-http-fixture"}]
        assert result["geometry"] == [{"dimensions": 6, "four_turns_exact": True}]
        assert response.headers["Content-Type"] == "application/json; charset=utf-8"
        assert response.headers["Cache-Control"] == "no-store"
    with urlopen(cube_http + "/api/results") as response:
        assert json.load(response)["experiments"] == []


def test_http_missing_cube_results_remain_explicit(cube_http, tmp_path):
    (tmp_path / "cube.json").unlink()
    with pytest.raises(HTTPError) as error:
        urlopen(cube_http + "/api/cube-results")
    assert error.value.code == 404
    assert "No recorded run" in json.load(error.value)["error"]


def test_http_unsupported_cube_result_schema_is_not_served_as_a_measurement(cube_http, tmp_path):
    (tmp_path / "cube.json").write_text('{"schema_version": 999}', encoding="utf-8")
    with pytest.raises(HTTPError) as error:
        urlopen(cube_http + "/api/cube-results")
    assert error.value.code == 500
    assert "unsupported schema" in json.load(error.value)["error"]
