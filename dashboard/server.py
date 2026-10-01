"""Serve recorded experiments and bounded, real codec inspections locally."""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import mimetypes
import ipaddress
import socket
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

STATIC_ROOT = Path(__file__).resolve().parent
MAX_BODY = 262_144
MAX_VALUES = 4096
MAX_DISCARD_BODY = 1024 * 1024
DISCARD_TIMEOUT = 1.0


def inspect_tensor(payload: dict) -> dict:
    """Encode a bounded input with the same codecs used by the experiments."""
    import torch
    from chromatensor.tensor import ChromaTensor
    from chromatensor.codecs import (
        HierarchicalCodec, HSVCodec, HueCodec, LearnedCodec, LearnedRGBDCodec,
        RGBCodec, RGBDCodec, UniformCodec,
    )

    values = payload.get("values")
    shape = payload.get("shape")
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_VALUES:
        raise ValueError(f"values must contain 1 to {MAX_VALUES} numerical items")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
        raise ValueError("values must be finite numbers")
    if not isinstance(shape, list) or not shape or not all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in shape):
        raise ValueError("shape must contain positive integers")
    if len(shape) > 8 or math.prod(shape) != len(values):
        raise ValueError("shape must match the number of values and have at most eight dimensions")
    if max(abs(v) for v in values) > 1e30:
        raise ValueError("values must stay within the float32 working range of this inspector")
    name = payload.get("codec", "rgbd")
    minimum, maximum = min(values), max(values)
    if minimum == maximum:
        minimum -= max(1.0, abs(minimum) * 0.01)
        maximum += max(1.0, abs(maximum) * 0.01)
    x = torch.tensor(values, dtype=torch.float32).reshape(shape)
    metadata = {"minimum": minimum, "maximum": maximum}
    if name == "rgb":
        codec = RGBCodec(minimum=minimum, maximum=maximum)
    elif name == "rgbd":
        codec = RGBDCodec()
        metadata = {"channels": ["R", "G", "B", "D"],
                    "D": "IEEE-754 biased exponent", "bits_per_value": 32}
    elif name == "hsv":
        codec = HSVCodec(minimum=minimum, maximum=maximum)
    elif name == "hue":
        period = float(payload.get("period", 2 * math.pi))
        if not math.isfinite(period) or period <= 0 or period > 1e30:
            raise ValueError("period must be positive and finite")
        codec = HueCodec(period=period)
        metadata = {"period": period, "semantics": "values wrap modulo period"}
    elif name in {"int8", "int4", "fixed24"}:
        bits = {"int8": 8, "int4": 4, "fixed24": 24}[name]
        codec = UniformCodec(bits=bits, minimum=minimum, maximum=maximum)
        metadata["bits"] = bits
    elif name == "learned":
        size = min(64, max(2, len(values)))
        codec = LearnedCodec(size=size, minimum=minimum, maximum=maximum).fit(x, seed=0)
        metadata.update({"vocabulary_size": size, "fit": "fitted on these input values, seed 0"})
    elif name == "learned_rgbd":
        size = min(64, max(2, len(values)))
        codec = LearnedRGBDCodec(size=size, modifier_size=16, minimum=minimum, maximum=maximum).fit(x, seed=0)
        metadata.update({"vocabulary_size": size, "modifier_size": 16, "fit": "fitted on these input values, seed 0"})
    elif name == "hierarchical":
        codec = HierarchicalCodec(levels=3, vocabulary_size=16, minimum=minimum, maximum=maximum)
        metadata.update({"levels": 3, "vocabulary_size": 16})
    else:
        raise ValueError("unknown codec")
    tensor = ChromaTensor.from_tensor(x, codec)
    decoded = tensor.decode().detach().cpu().reshape(-1).tolist()
    rgb = tensor.rgb.detach().cpu().reshape(-1, 3).tolist()
    codes = tensor.codes.detach().cpu().reshape(len(values), -1).tolist()
    tokens = ["/".join(str(int(c)) for c in code) for code in codes]
    frequencies = Counter(tokens)
    errors = [abs(float(v) - float(d)) for v, d in zip(values, decoded)]
    if name == "hue":
        period = metadata["period"]
        errors = [min(e % period, period - e % period) for e in errors]
    hsv = [list(colorsys.rgb_to_hsv(*(float(c) / 255 for c in color))) for color in rgb]
    return {
        "name": "Interactive input", "shape": shape, "values": values,
        "decoded": decoded, "rgb": rgb, "hsv": hsv, "tokens": tokens,
        "codes": codes, "codec": name, "metadata": metadata,
        "frequencies": [frequencies[t] for t in tokens], "errors": errors,
        "metrics": {"mae": sum(errors) / len(errors), "max_error": max(errors)},
        "codec_config": codec.to_config(),
        "provenance": "Encoded by the installed ChromaTensor Python codecs on this server",
    }


class DashboardServer(ThreadingHTTPServer):
    """Threaded read-only result server, with one bounded codec endpoint."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], results_path: Path | str | None = None):
        self.results_path = Path(results_path or Path.cwd() / "results" / "latest.json").resolve()
        self.workspace_root = self.results_path.parent.parent if self.results_path.parent.name == "results" else Path.cwd().resolve()
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(address, DashboardHandler)
        host = address[0].lower()
        try:
            ip = ipaddress.ip_address(host)
            self.allowed_hosts = {ip.compressed}
            if ip.is_loopback:
                self.allowed_hosts.update({"localhost", "127.0.0.1", "::1"})
        except ValueError:
            self.allowed_hosts = {host, self.server_address[0].lower()}
            if host == "localhost":
                self.allowed_hosts.update({"127.0.0.1", "::1"})


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardServer

    def log_message(self, fmt, *args):
        return None

    def _send(self, status: int, body: bytes, mime: str):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, data: dict):
        self._send(status, json.dumps(data, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_HEAD(self):
        self.do_GET()

    def _authority(self, value, *, origin=False):
        """Parse one exact HTTP authority; credentials and alternate URLs fail."""
        if not isinstance(value, str) or not value or any(char.isspace() for char in value):
            raise ValueError("Invalid local authority")
        parsed = urlsplit(value if origin else "//" + value)
        if (parsed.username is not None or parsed.password is not None
                or parsed.path or parsed.query or parsed.fragment or not parsed.hostname
                or origin and parsed.scheme != "http"):
            raise ValueError("Invalid local authority")
        host = parsed.hostname.lower()
        try:
            host = ipaddress.ip_address(host).compressed
        except ValueError:
            pass
        return host, parsed.port if parsed.port is not None else 80

    def _local_request(self):
        try:
            hosts = self.headers.get_all("Host", [])
            if len(hosts) != 1:
                return False
            host, port = self._authority(hosts[0])
            return host in self.server.allowed_hosts and port == self.server.server_port
        except ValueError:
            return False

    def _same_origin(self):
        origins = self.headers.get_all("Origin", [])
        if not origins:
            return True
        try:
            return len(origins) == 1 and self._authority(origins[0], origin=True) == self._authority(self.headers["Host"])
        except ValueError:
            return False

    def _reject_post(self, status, error):
        # Closing a Windows socket with unread POST bytes can reset the TCP
        # connection before the client receives its HTTP rejection. Discard
        # only a bounded, explicitly sized body, without parsing or executing it.
        self.close_connection = True
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) == 1 and not self.headers.get_all("Transfer-Encoding", []):
            try:
                length = int(lengths[0])
                if 0 < length <= MAX_DISCARD_BODY:
                    deadline = time.monotonic() + DISCARD_TIMEOUT
                    previous_timeout = self.connection.gettimeout()
                    try:
                        while length:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                break
                            self.connection.settimeout(remaining)
                            block = self.rfile.read1(min(length, 65536))
                            if not block:
                                break
                            length -= len(block)
                    finally:
                        self.connection.settimeout(previous_timeout)
            except (ValueError, OSError):
                pass
        return self._json(status, {"error": error})

    def do_GET(self):
        if not self._local_request():
            return self._json(421, {"error": "Use an allowed dashboard hostname and port."})
        request = urlsplit(self.path)
        if request.path == "/api/health":
            return self._json(200, {"status": "ok", "results_available": self.server.results_path.is_file(), "service": "ChromaTensor local dashboard"})
        if request.path in ("/api/results", "/api/cube-results"):
            results_path = self.server.results_path if request.path == "/api/results" else self.server.results_path.parent / "cube.json"
            if not results_path.is_file():
                command = "scripts/run_benchmarks.py" if request.path == "/api/results" else "scripts/run_cube_experiments.py"
                return self._json(404, {"error": f"No recorded run yet. Run python {command}, then refresh this page."})
            try:
                data = json.loads(results_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or data.get("schema_version") != 1:
                    raise ValueError("unsupported result schema")
                return self._json(200, data)
            except (OSError, ValueError, TypeError):
                return self._json(500, {"error": "The recorded results file is unreadable or uses an unsupported schema."})
        if request.path == "/api/artifact":
            relative = parse_qs(request.query).get("path", [""])[0]
            allowed_roots = [self.server.results_path.parent.resolve(), (self.server.workspace_root / "artifacts").resolve()]
            target = (self.server.workspace_root / relative).resolve()
            if not any(target.is_relative_to(base) for base in allowed_roots) or not target.is_file() or target.suffix not in {".json", ".csv", ".txt", ".md", ".ctensor", ".ctcheckpoint", ".npz", ".pt", ".png", ".jpg", ".jpeg"}:
                return self._json(404, {"error": "Artifact not found in the results or artifacts directory."})
            try:
                return self._send(200, target.read_bytes(), mimetypes.guess_type(target.name)[0] or "application/octet-stream")
            except OSError:
                return self._json(404, {"error": "Artifact could not be read."})
        static_name = {"/": "index.html", "/index.html": "index.html", "/style.css": "style.css", "/app.js": "app.js", "/cube.js": "cube.js", "/cube.css": "cube.css"}.get(request.path)
        if static_name:
            target = STATIC_ROOT / static_name
            return self._send(200, target.read_bytes(), (mimetypes.guess_type(static_name)[0] or "text/plain") + "; charset=utf-8")
        return self._json(404, {"error": "Not found"})

    def do_POST(self):
        if not self._local_request():
            return self._reject_post(421, "Use an allowed dashboard hostname and port.")
        endpoint = urlsplit(self.path).path
        if endpoint not in ("/api/tensor", "/api/cube"):
            return self._reject_post(404, "Not found")
        # Reject cross-site requests, including browser requests to a localhost server.
        if not self._same_origin():
            return self._reject_post(403, "Use this endpoint from the local dashboard.")
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            return self._reject_post(415, "Use application/json")
        try:
            size = int(self.headers.get("Content-Length", "0"))
            maximum_body = 1024 * 1024 if endpoint == "/api/cube" else MAX_BODY
            if not 0 < size <= maximum_body:
                return self._reject_post(413, "Request body exceeds the inspector limit.")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("Request must be a JSON object")
            if endpoint == "/api/cube":
                from .cube_api import inspect_cube
                result = inspect_cube(payload, self.server.results_path)
            else:
                result = inspect_tensor(payload)
            return self._json(200, result)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            return self._json(400, {"error": str(exc)})
        except ImportError:
            return self._json(503, {"error": "Install ChromaTensor and PyTorch before using the live inspector."})
        except Exception:
            return self._json(500, {"error": "The codec could not encode this input. Check the server environment."})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address, default localhost")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--results", type=Path, default=Path.cwd() / "results" / "latest.json")
    args = parser.parse_args(argv)
    server = DashboardServer((args.host, args.port), args.results)
    print(f"ChromaTensor dashboard: http://{args.host}:{server.server_port}", flush=True)
    print(f"Results: {server.results_path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("ChromaTensor dashboard stopped.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

