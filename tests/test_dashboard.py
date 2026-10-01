"""Exercise the local dashboard's real codec path and HTTP boundaries."""

import json
import http.client
import socket
import threading
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pytest

from dashboard.server import DashboardHandler, DashboardServer, MAX_BODY, inspect_tensor


@pytest.fixture
def dashboard_http(tmp_path):
    results = tmp_path / "latest.json"
    results.write_text(json.dumps({"schema_version": 1, "run_id": "test-run", "experiments": []}), encoding="utf-8")
    server = DashboardServer(("127.0.0.1", 0), results)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_port}", results
    server.shutdown()
    server.server_close()
    worker.join(timeout=3)


@pytest.mark.parametrize("codec", ["rgb", "rgbd", "hsv", "hue", "int8", "int4", "fixed24", "learned", "learned_rgbd", "hierarchical"])
def test_tensor_inspector_uses_real_codecs(codec):
    result = inspect_tensor({"values": [-1.0, -0.2, 0.2, 1.0], "shape": [2, 2], "codec": codec})
    assert result["shape"] == [2, 2]
    assert len(result["rgb"]) == 4
    assert all(len(rgb) == 3 and all(0 <= c <= 255 for c in rgb) for rgb in result["rgb"])
    assert len(result["decoded"]) == len(result["tokens"]) == 4
    assert result["metrics"]["mae"] >= 0
    assert result["metrics"]["max_error"] >= result["metrics"]["mae"]
    assert all(frequency >= 1 for frequency in result["frequencies"])
    if codec in {"rgb", "rgbd", "fixed24"}:
        assert result["metrics"]["max_error"] < 1e-6


def test_inspector_default_stores_actual_rgbd_channels_and_decodes_them():
    result = inspect_tensor({"values": [1.0, -1.0, 1.5, 2.0], "shape": [2, 2]})
    assert result["codec"] == "rgbd"
    assert result["codes"] == [[0, 0, 0, 127], [128, 0, 0, 127], [64, 0, 0, 127], [0, 0, 0, 128]]
    assert result["decoded"] == [1.0, -1.0, 1.5, 2.0]
    assert result["rgb"] == [row[:3] for row in result["codes"]]
    assert result["metadata"]["channels"] == ["R", "G", "B", "D"]
    assert result["metrics"]["max_error"] == 0


@pytest.mark.parametrize("payload", [
    {"values": [], "shape": [1]},
    {"values": [float("nan")], "shape": [1]},
    {"values": [float("inf")], "shape": [1]},
    {"values": [True], "shape": [1]},
    {"values": [1], "shape": [2]},
    {"values": [1], "shape": [True]},
    {"values": [1], "shape": [0]},
    {"values": [1], "shape": [1], "codec": "unrecognized"},
    {"values": [1], "shape": [1], "codec": "hue", "period": -1},
    {"values": [1] * 4097, "shape": [4097]},
])
def test_inspector_rejects_invalid_or_unbounded_requests(payload):
    with pytest.raises(ValueError):
        inspect_tensor(payload)


def test_http_serves_measured_results_and_static_assets(dashboard_http):
    base, _ = dashboard_http
    with urlopen(base + "/api/results") as response:
        result = json.load(response)
        assert result["run_id"] == "test-run"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
    with urlopen(base + "/") as response:
        html = response.read().decode()
        assert '<html lang="en">' in html
        assert 'id="content"' in html
    with urlopen(base + "/app.js") as response:
        assert "javascript" in response.headers["Content-Type"]
    with urlopen(base + "/api/health") as response:
        assert json.load(response)["results_available"] is True


def test_missing_results_remain_explicit(dashboard_http):
    base, results = dashboard_http
    results.unlink()
    with pytest.raises(HTTPError) as error:
        urlopen(base + "/api/results")
    assert error.value.code == 404
    assert "No recorded run" in json.load(error.value)["error"]


def test_invalid_results_cannot_be_presented_as_measurements(dashboard_http):
    base, results = dashboard_http
    results.write_text('{"schema_version": 999}', encoding="utf-8")
    with pytest.raises(HTTPError) as error:
        urlopen(base + "/api/results")
    assert error.value.code == 500


def test_http_tensor_endpoint_matches_direct_codec_result(dashboard_http):
    base, _ = dashboard_http
    payload = {"values": [-0.5, 0.5], "shape": [1, 2], "codec": "rgb"}
    request = Request(base + "/api/tensor", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request) as response:
        assert json.load(response) == inspect_tensor(payload)


def test_http_rejects_cross_origin_tensor_requests(dashboard_http):
    base, _ = dashboard_http
    request = Request(base + "/api/tensor", data=b"{}", headers={"Content-Type": "application/json", "Origin": "https://example.com"})
    with pytest.raises(HTTPError) as error:
        urlopen(request)
    assert error.value.code == 403


@pytest.mark.parametrize("path", ["/", "/api/health", "/api/results", "/api/tensor", "/api/cube"])
def test_foreign_host_cannot_read_or_compute_even_with_matching_origin(dashboard_http, path):
    base, _ = dashboard_http
    body = b"{}" if path in {"/api/tensor", "/api/cube"} else None
    request = Request(base + path, data=body, headers={"Host": "attacker.example", "Origin": "http://attacker.example", "Content-Type": "application/json"})
    with pytest.raises(HTTPError) as error:
        urlopen(request)
    assert error.value.code == 421


@pytest.mark.parametrize("path", ["/api/tensor", "/api/cube"])
def test_rejected_post_body_can_arrive_after_headers_without_reset(dashboard_http, monkeypatch, path):
    base, _ = dashboard_http
    checked = threading.Event()
    local_request = DashboardHandler._local_request
    def observe_host_check(handler):
        allowed = local_request(handler)
        if not allowed:
            checked.set()
        return allowed
    monkeypatch.setattr(DashboardHandler, "_local_request", observe_host_check)
    def unexpected_execution(*args, **kwargs):
        pytest.fail("A rejected body reached the codec endpoint")
    monkeypatch.setattr("dashboard.server.inspect_tensor", unexpected_execution)
    monkeypatch.setattr("dashboard.cube_api.inspect_cube", unexpected_execution)
    body = b"x" * MAX_BODY
    for _ in range(12):
        checked.clear()
        connection = http.client.HTTPConnection("127.0.0.1", int(base.rsplit(":", 1)[1]), timeout=5)
        try:
            connection.putrequest("POST", path, skip_host=True)
            connection.putheader("Host", "attacker.example")
            connection.putheader("Origin", "http://attacker.example")
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", str(len(body)))
            connection.endheaders()
            # Deliver the body only after the server has inspected the headers.
            assert checked.wait(2)
            connection.send(body[:1])
            connection.send(body[1:])
            response = connection.getresponse()
            assert response.status == 421
            assert "allowed dashboard hostname" in json.loads(response.read())["error"]
        finally:
            connection.close()


@pytest.mark.parametrize("declared_length", [0, 2 ** 30, "invalid"])
def test_foreign_host_denial_does_not_wait_for_unbounded_or_invalid_body(dashboard_http, declared_length):
    base, _ = dashboard_http
    connection = http.client.HTTPConnection("127.0.0.1", int(base.rsplit(":", 1)[1]), timeout=2)
    try:
        connection.putrequest("POST", "/api/cube", skip_host=True)
        connection.putheader("Host", "attacker.example")
        connection.putheader("Content-Length", str(declared_length))
        connection.endheaders()
        response = connection.getresponse()
        assert response.status == 421
        response.read()
    finally:
        connection.close()


def test_foreign_host_denial_bounds_wait_for_an_unfinished_body(dashboard_http):
    base, _ = dashboard_http
    connection = http.client.HTTPConnection("127.0.0.1", int(base.rsplit(":", 1)[1]), timeout=3)
    try:
        connection.putrequest("POST", "/api/cube", skip_host=True)
        connection.putheader("Host", "attacker.example")
        connection.putheader("Content-Length", "100")
        connection.endheaders()
        response = connection.getresponse()
        assert response.status == 421
        response.read()
    finally:
        connection.close()


@pytest.mark.parametrize("origin", ["https://127.0.0.1:{port}", "http://localhost:{port}", "http://127.0.0.1:1", "null", "http://user@127.0.0.1:{port}", "http://127.0.0.1:{port}/", ""])
def test_origin_requires_exact_http_authority(dashboard_http, origin):
    base, _ = dashboard_http
    port = base.rsplit(":", 1)[1]
    request = Request(base + "/api/tensor", data=b"{}", headers={"Content-Type": "application/json", "Origin": origin.format(port=port)})
    with pytest.raises(HTTPError) as error:
        urlopen(request)
    assert error.value.code == 403


def test_localhost_alias_with_its_exact_origin_is_allowed(dashboard_http):
    base, _ = dashboard_http
    host = "localhost:" + base.rsplit(":", 1)[1]
    request = Request(base + "/api/tensor", data=json.dumps({"values": [0, 1], "shape": [2]}).encode(), headers={"Host": host, "Origin": "http://" + host, "Content-Type": "application/json"})
    with urlopen(request) as response:
        assert response.status == 200


def test_ipv6_loopback_requires_bracketed_authority_and_matching_origin(tmp_path):
    if not socket.has_ipv6:
        pytest.skip("Host has no IPv6 support")
    results = tmp_path / "latest.json"
    results.write_text('{"schema_version":1,"experiments":[]}')
    try:
        server = DashboardServer(("::1", 0), results)
    except OSError:
        pytest.skip("IPv6 loopback is unavailable")
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        host = f"[::1]:{server.server_port}"
        connection = http.client.HTTPConnection("::1", server.server_port, timeout=5)
        connection.request("POST", "/api/tensor", body=json.dumps({"values": [0, 1], "shape": [2]}),
                           headers={"Host": host, "Origin": "http://" + host, "Content-Type": "application/json"})
        response = connection.getresponse()
        assert response.status == 200
        response.read()
        connection.close()
        connection = http.client.HTTPConnection("::1", server.server_port, timeout=5)
        connection.request("GET", "/api/results", headers={"Host": f"attacker.example:{server.server_port}"})
        response = connection.getresponse()
        assert response.status == 421
        response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def test_http_reads_an_allowed_absolute_artifact_path(dashboard_http):
    base, results = dashboard_http
    artifact = results.parent / "measured.json"
    artifact.write_text('{"measured": true}', encoding="utf-8")
    with urlopen(base + "/api/artifact?path=" + quote(str(artifact))) as response:
        assert json.load(response) == {"measured": True}


@pytest.mark.parametrize("path", ["/../pyproject.toml", "/%2e%2e/pyproject.toml", "/api/artifact?path=pyproject.toml", "/api/artifact?path=../chromatic/pyproject.toml"])
def test_http_does_not_serve_paths_outside_results(dashboard_http, path):
    base, _ = dashboard_http
    with pytest.raises(HTTPError) as error:
        urlopen(base + path)
    assert error.value.code == 404
