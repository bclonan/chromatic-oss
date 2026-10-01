"""Verify actual RGBD tokens against IEEE bits, then decode in a new process."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from chromatensor.tokenizer import ChromaDecoder, ChromaTokenizer, TokenSequence
from experiments.common import environment, write_json
from experiments.provenance import artifact_manifest, capture_provenance, file_hash, unique_run_id, verify_provenance

KNOWN = [(0x00000000, [0, 0, 0, 0]), (0x80000000, [128, 0, 0, 0]),
         (0x3F800000, [0, 0, 0, 127]), (0xBF800000, [128, 0, 0, 127]),
         (0x3FC00000, [64, 0, 0, 127]), (0x40000000, [0, 0, 0, 128]),
         (0x7F7FFFFF, [127, 255, 255, 254]), (0x00000001, [0, 0, 1, 0]),
         (0x80000001, [128, 0, 1, 0])]
CHECKS = ("independent_rgbd_byte_oracle", "exact_fp32_bits", "every_channel_changes_decode",
          "depth_distinguishes_equal_rgb", "file_payload_is_actual_rgbd", "no_original_float_metadata",
          "cold_process_file_only_decode")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def independent_rgbd(bits):
    """Derive bytes with NumPy masks, without calling library codec helpers."""
    bits = np.asarray(bits, dtype=np.uint32)
    return np.stack((((bits >> 24) & 128) | ((bits >> 16) & 127),
                     (bits >> 8) & 255, bits & 255, (bits >> 23) & 255), axis=-1).astype(np.uint8)


def checked_file(path):
    """Read the actual native file bytes independently of the package loader."""
    data = Path(path).read_bytes()
    if data[:8] != b"CTENSOR\x01" or sha256(data[:-32]) != data[-32:].hex():
        raise ValueError("Native token file magic or checksum differs")
    size = struct.unpack("<I", data[8:12])[0]
    metadata = json.loads(data[12:12 + size])
    payload = data[12 + size:-32]
    if metadata["compression"] != "none":
        raise ValueError("The byte proof requires an uncompressed RGBD file")
    if metadata["code_dtype"] != "uint8" or metadata["codec"] != {
            "name": "rgbd", "format": "finite-ieee754-fp32-sign-fraction-exponent"}:
        raise ValueError("Native file does not store finite RGBD byte tokens")
    if metadata["raw_bytes"] != len(payload) or metadata["payload_bytes"] != len(payload):
        raise ValueError("Native token byte counts differ")
    if metadata["code_shape"] != [*metadata["shape"], 4]:
        raise ValueError("Native file has no fourth depth channel")
    return metadata, payload, size, len(data)


def decode_saved_file(path, expected_sha256):
    """The child receives only a file path and a reference digest, never inputs."""
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise ValueError("Expected decoded SHA-256 must be lowercase hexadecimal")
    path = Path(path).resolve()
    metadata, payload, header_bytes, file_bytes = checked_file(path)
    sequence = TokenSequence.load(path, max_file_bytes=file_bytes,
                                  max_header_bytes=header_bytes, max_payload_bytes=len(payload))
    if sequence.tokens.detach().cpu().contiguous().numpy().tobytes() != payload:
        raise ValueError("Loaded tokens differ from actual file payload")
    decoded = ChromaDecoder().decode(sequence)
    decoded_bytes = decoded.detach().cpu().contiguous().numpy().tobytes()
    actual = sha256(decoded_bytes)
    receipt = {"pid": os.getpid(), "python": sys.executable,
               "input_arguments": ["saved_file", "expected_decoded_sha256"],
               "file": str(path), "file_sha256": file_hash(path), "file_bytes": file_bytes,
               "payload_bytes": len(payload), "header_bytes": header_bytes,
               "shape": list(sequence.shape), "token_shape": list(sequence.tokens.shape),
               "token_dtype": str(sequence.tokens.dtype), "num_tokens": sequence.num_tokens,
               "decoded_dtype": str(decoded.dtype), "decoded_sha256": actual,
               "expected_decoded_sha256": expected_sha256, "exact_sha256_match": actual == expected_sha256,
               "payload_sha256": sha256(payload), "header": metadata}
    if not receipt["exact_sha256_match"]:
        raise ValueError("Saved RGBD tokens do not reconstruct the reference FP32 bytes")
    return receipt


def random_finite_bits(count, seed):
    """Sample uint32 patterns uniformly, discarding all exponent-255 patterns."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    chunks, remaining, candidates = [], count, 0
    while remaining:
        size = max(remaining + remaining // 64 + 32, 64)
        raw = torch.randint(0, 2 ** 32, (size,), dtype=torch.int64, generator=generator).numpy()
        candidates += size
        finite = raw[((raw >> 23) & 255) != 255][:remaining].astype(np.uint32)
        chunks.append(finite)
        remaining -= finite.size
    bits = np.concatenate(chunks)
    for index, item in enumerate(KNOWN[:count]):
        bits[index] = item[0]
    return bits, candidates


def sampling_config(seed, samples, threads):
    return {"seed": seed, "sample_count": samples, "threads": threads,
            "device": "cpu", "execution_dtype": "torch.float32", "token_dtype": "torch.uint8",
            "token_channels": ["R", "G", "B", "D"], "tokens_per_value": 1,
            "dataset": "seeded uint32 bit patterns filtered to finite IEEE-754 FP32",
            "known_bit_patterns": [f"{item[0]:08x}" for item in KNOWN],
            "oracle": "independent NumPy integer masks", "file_compression": "none",
            "cold_process_inputs": ["saved_file", "expected_decoded_sha256"]}


def validate_rgbd_evidence(record, root=ROOT, *, require_release=True):
    """Recompute stored bytes and decoded bits. Claimed check flags are insufficient."""
    root = Path(root).resolve()
    if not isinstance(record, dict) or record.get("schema_version") != 1 or record.get("status") != "complete":
        raise ValueError("RGBD evidence is missing or incomplete")
    config = record.get("config")
    if not isinstance(config, dict):
        raise ValueError("RGBD evidence configuration is missing")
    count, seed, threads = config.get("sample_count"), config.get("seed"), config.get("threads")
    minimum = 1_000_000 if require_release else len(KNOWN)
    if type(count) is not int or count < minimum:
        raise ValueError("RGBD evidence has insufficient real sample count")
    if count > 100_000_000:
        raise ValueError("RGBD evidence sample count exceeds the validation resource budget")
    if type(seed) is not int or not -(2 ** 63) <= seed < 2 ** 64 or type(threads) is not int or threads < 1:
        raise ValueError("RGBD evidence seed or thread count is invalid")
    if config != sampling_config(seed, count, threads):
        raise ValueError("RGBD evidence configuration differs from the byte proof contract")
    storage, cold = record.get("storage"), record.get("cold_process")
    if not isinstance(storage, dict) or not isinstance(cold, dict):
        raise ValueError("RGBD evidence has no persisted file and cold process receipt")
    paths = [storage.get("path"), cold.get("receipt")]
    if any(not isinstance(path, str) for path in paths):
        raise ValueError("RGBD evidence artifact paths are missing")
    verify_provenance(record.get("provenance"), config, root, required_artifacts=paths)
    resolved = [(root / path).resolve() for path in paths]
    if any(not path.is_relative_to(root) for path in resolved):
        raise ValueError("RGBD evidence artifact escaped the workspace")
    token_file, receipt_file = resolved
    if token_file.parent != receipt_file.parent or token_file.parent.name != record.get("run_id"):
        raise ValueError("RGBD files must share their immutable run directory")
    if record.get("checks") != {name: True for name in CHECKS}:
        raise ValueError("RGBD evidence check labels are missing or incomplete")
    metadata, payload, header_bytes, file_bytes = checked_file(token_file)
    if (metadata["shape"] != [count] or metadata["code_shape"] != [count, 4]
            or metadata["tensor_dtype"] != "torch.float32" or len(payload) != count * 4):
        raise ValueError("RGBD file sample count, shape, dtype or byte count differs")
    allowed = {"version", "shape", "code_shape", "code_dtype", "tensor_dtype", "codec", "compression",
               "raw_bytes", "payload_bytes"}
    if set(metadata) != allowed:
        raise ValueError("RGBD file contains metadata beyond its decode contract")
    expected_storage = {"file_bytes": file_bytes, "payload_bytes": len(payload), "header_bytes": header_bytes,
                        "container_bytes": file_bytes - len(payload) - header_bytes, "bits_per_value_payload": 32,
                        "bits_per_value_total": file_bytes * 8 / count, "header": metadata,
                        "payload_sha256": sha256(payload)}
    if any(storage.get(key) != value for key, value in expected_storage.items()):
        raise ValueError("RGBD measured file byte counts or hashes differ")
    bits, candidates = random_finite_bits(count, seed)
    oracle = independent_rgbd(bits)
    if payload != oracle.tobytes():
        raise ValueError("Saved RGBD bytes differ from the independent seeded NumPy oracle")
    expected_sha = sha256(bits.astype("<u4", copy=False).tobytes())
    sequence = TokenSequence.load(token_file, max_file_bytes=file_bytes, max_header_bytes=header_bytes,
                                  max_payload_bytes=len(payload))
    if sequence.tokens.numpy().tobytes() != payload:
        raise ValueError("Loaded RGBD rows differ from the saved byte payload")
    decoded = ChromaDecoder().decode(sequence)
    reconstructed = decoded.contiguous().numpy().view(np.uint32)
    mismatches = int(np.count_nonzero(reconstructed != bits))
    if mismatches or sha256(decoded.contiguous().numpy().tobytes()) != expected_sha:
        raise ValueError("Reloaded RGBD bytes do not exactly reconstruct the seeded FP32 bits")
    encoding, decoding = record.get("encoding"), record.get("decoding")
    if not isinstance(encoding, dict) or not isinstance(decoding, dict):
        raise ValueError("RGBD encoding or decoding measurements are absent")
    known_records = [{"fp32_bits_hex": f"{pattern:08x}", "rgbd": expected} for pattern, expected in KNOWN]
    expected_encoding = {"samples": count, "candidate_patterns": candidates, "finite_exponent_pattern_filter": "D != 255",
                         "known_patterns": known_records, "token_shape": [count, 4], "oracle_byte_mismatches": 0}
    if any(encoding.get(key) != value for key, value in expected_encoding.items()):
        raise ValueError("RGBD recorded encoding measurements differ from actual tokens")
    if (decoding.get("bit_mismatches") != mismatches or decoding.get("expected_fp32_sha256") != expected_sha
            or decoding.get("decoded_fp32_sha256") != expected_sha):
        raise ValueError("RGBD recorded decoded bits or digests differ")
    mutation_records = decoding.get("channel_mutations")
    if not isinstance(mutation_records, list) or len(mutation_records) != 4:
        raise ValueError("RGBD evidence must prove use of all four token channels")
    for channel, label in enumerate(("R", "G", "B", "D")):
        token = torch.tensor([[0, 0, 0, 127]], dtype=torch.uint8)
        token[0, channel] ^= 1
        after = ChromaDecoder().decode(TokenSequence(token, (1,), sequence.codec_config))
        actual = int(after.numpy().view(np.uint32)[0])
        expected = (0x3F800000 ^ (1 << (16 - channel * 8))) if channel < 3 else 0x3F000000
        measured = {"channel": label, "before_rgbd": [0, 0, 0, 127], "after_rgbd": token[0].tolist(),
                    "before_bits_hex": "3f800000", "after_bits_hex": f"{actual:08x}",
                    "expected_bits_hex": f"{expected:08x}", "decoded_value": float(after[0]), "used_by_decoder": True}
        if actual != expected or mutation_records[channel] != measured:
            raise ValueError("RGBD channel mutation measurement differs from actual decoding")
    equal_rgb = sequence.tokens[2, :3].tolist() == sequence.tokens[5, :3].tolist()
    different_d = sequence.tokens[2, 3].item() != sequence.tokens[5, 3].item()
    expected_depth = {"value_one": sequence.tokens[2].tolist(), "value_two": sequence.tokens[5].tolist(),
                      "same_rgb": equal_rgb, "different_d": different_d}
    if not equal_rgb or not different_d or record.get("depth_required") != expected_depth:
        raise ValueError("RGBD depth does not distinguish the required equal-RGB values")
    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
    expected_cold = {"input_arguments": ["saved_file", "expected_decoded_sha256"],
                     "file_sha256": file_hash(token_file), "file_bytes": file_bytes,
                     "payload_bytes": len(payload), "header_bytes": header_bytes, "shape": [count],
                     "token_shape": [count, 4], "token_dtype": "torch.uint8", "num_tokens": count,
                     "decoded_dtype": "torch.float32", "decoded_sha256": expected_sha,
                     "expected_decoded_sha256": expected_sha, "exact_sha256_match": True,
                     "payload_sha256": sha256(payload), "header": metadata}
    if not isinstance(receipt, dict) or any(receipt.get(key) != value for key, value in expected_cold.items()):
        raise ValueError("RGBD cold-process receipt differs from the actual saved tokens")
    if (type(receipt.get("pid")) is not int or receipt["pid"] <= 0
            or type(record.get("encoding_pid")) is not int or record["encoding_pid"] <= 0
            or receipt["pid"] == record["encoding_pid"] or cold.get("pid") != receipt["pid"]
            or cold.get("returncode") != 0 or cold.get("exact_sha256_match") is not True
            or cold.get("decoded_sha256") != expected_sha):
        raise ValueError("RGBD cold decoding did not use a separate successful process")
    for duration in (encoding.get("seconds"), decoding.get("seconds"), cold.get("seconds"), record.get("total_seconds")):
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("RGBD execution timings are missing or invalid")
    return {"verified": True, "run_id": record["run_id"], "samples": count,
            "token_shape": [count, 4], "token_dtype": "torch.uint8", "bit_mismatches": mismatches,
            "channels_used": ["R", "G", "B", "D"], "payload_bytes": len(payload), "file_bytes": file_bytes,
            "decoded_sha256": expected_sha, "cold_process_pid": receipt["pid"],
            "source_sha256": record["provenance"]["source"]["sha256"]}


def verify(args):
    if args.samples < len(KNOWN):
        raise ValueError("At least nine samples are required for the known-bit checks")
    if args.threads < 1:
        raise ValueError("Threads must be positive")
    torch.set_num_threads(args.threads)
    config = sampling_config(args.seed, args.samples, args.threads)
    run_id = unique_run_id("rgbd-tokens")
    directory = ROOT / "artifacts" / "rgbd-tokens" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    token_file, child_file = directory / "tokens.ctensor", directory / "cold-decode.json"
    immutable_result = directory / "verification.json"
    provenance = capture_provenance(config, ROOT)
    started = time.perf_counter_ns()
    try:
        bits, candidates = random_finite_bits(args.samples, args.seed)
        values = torch.from_numpy(bits.view(np.float32).copy())
        expected_hash = sha256(bits.astype("<u4", copy=False).tobytes())
        before = time.perf_counter_ns()
        sequence = ChromaTokenizer().encode(values)
        encode_seconds = (time.perf_counter_ns() - before) / 1e9
        expected_tokens = independent_rgbd(bits)
        if not np.array_equal(sequence.tokens.numpy(), expected_tokens):
            raise ValueError("RGBD tokens differ from the independent NumPy byte oracle")
        known_records = []
        for index, (pattern, expected) in enumerate(KNOWN):
            actual = sequence.tokens[index].tolist()
            if actual != expected:
                raise ValueError("A known FP32 bit pattern has incorrect RGBD bytes")
            known_records.append({"fp32_bits_hex": f"{pattern:08x}", "rgbd": actual})
        before = time.perf_counter_ns()
        decoded = ChromaDecoder().decode(sequence)
        decode_seconds = (time.perf_counter_ns() - before) / 1e9
        actual_bits = decoded.contiguous().numpy().view(np.uint32)
        mismatches = int(np.count_nonzero(actual_bits != bits))
        if mismatches:
            raise ValueError("RGBD roundtrip changed FP32 bits")
        decoded_hash = sha256(decoded.contiguous().numpy().tobytes())
        mutation_records = []
        for channel, label in enumerate(("R", "G", "B", "D")):
            token = torch.tensor([[0, 0, 0, 127]], dtype=torch.uint8)
            token[0, channel] ^= 1
            modified = TokenSequence(token, (1,), sequence.codec_config)
            reconstructed = ChromaDecoder().decode(modified)
            actual = int(reconstructed.numpy().view(np.uint32)[0])
            expected = (0x3F800000 ^ (1 << (16 - channel * 8))) if channel < 3 else 0x3F000000
            if actual != expected or actual == 0x3F800000:
                raise ValueError("A stored RGBD channel is not used by the decoder")
            mutation_records.append({"channel": label, "before_rgbd": [0, 0, 0, 127],
                                     "after_rgbd": token[0].tolist(), "before_bits_hex": "3f800000",
                                     "after_bits_hex": f"{actual:08x}", "expected_bits_hex": f"{expected:08x}",
                                     "decoded_value": float(reconstructed[0]), "used_by_decoder": True})
        written = sequence.save(token_file, compress=False)
        metadata, payload, header_bytes, file_bytes = checked_file(token_file)
        if payload != expected_tokens.tobytes() or len(payload) != args.samples * 4 or written != file_bytes:
            raise ValueError("Persisted file does not contain exactly four RGBD bytes per value")
        allowed_metadata = {"version", "shape", "code_shape", "code_dtype", "tensor_dtype", "codec",
                            "compression", "raw_bytes", "payload_bytes"}
        if set(metadata) != allowed_metadata:
            raise ValueError("Token file includes metadata beyond the decode contract")
        del values, decoded, actual_bits, bits, expected_tokens, sequence
        command = [sys.executable, str(Path(__file__).resolve()), "--decode-file", str(token_file),
                   "--expected-sha256", expected_hash]
        before = time.perf_counter_ns()
        child = subprocess.run(command, cwd=directory, capture_output=True, text=True,
                               check=True, timeout=args.child_timeout)
        cold_seconds = (time.perf_counter_ns() - before) / 1e9
        cold_receipt = json.loads(child.stdout)
        if cold_receipt["pid"] == os.getpid():
            raise ValueError("Cold decoding reused the encoding process")
        write_json(child_file, cold_receipt)
        artifacts = artifact_manifest([token_file, child_file], ROOT)
        provenance["artifacts"] = artifacts
        verify_provenance(provenance, config, ROOT, required_artifacts=[token_file, child_file])
        result = {"schema_version": 1, "status": "complete", "run_id": run_id,
                  "encoding_pid": os.getpid(),
                  "completed_at": datetime.now(timezone.utc).isoformat(), "config": config,
                  "environment": environment(), "provenance": provenance,
                  "encoding": {"samples": args.samples, "candidate_patterns": candidates,
                               "finite_exponent_pattern_filter": "D != 255", "known_patterns": known_records,
                               "token_shape": [args.samples, 4], "oracle_byte_mismatches": 0,
                               "seconds": encode_seconds},
                  "decoding": {"bit_mismatches": mismatches, "expected_fp32_sha256": expected_hash,
                               "decoded_fp32_sha256": decoded_hash, "seconds": decode_seconds,
                               "channel_mutations": mutation_records},
                  "depth_required": {"value_one": [0, 0, 0, 127], "value_two": [0, 0, 0, 128],
                                     "same_rgb": True, "different_d": True},
                  "storage": {"path": token_file.relative_to(ROOT).as_posix(), "file_bytes": file_bytes,
                              "payload_bytes": len(payload), "header_bytes": header_bytes,
                              "container_bytes": file_bytes - len(payload) - header_bytes,
                              "bits_per_value_payload": 32,
                              "bits_per_value_total": file_bytes * 8 / args.samples,
                              "header": metadata, "payload_sha256": sha256(payload)},
                  "cold_process": {"returncode": child.returncode, "seconds": cold_seconds,
                                   "receipt": child_file.relative_to(ROOT).as_posix(),
                                   "pid": cold_receipt["pid"], "exact_sha256_match": cold_receipt["exact_sha256_match"],
                                   "decoded_sha256": cold_receipt["decoded_sha256"], "stderr": child.stderr},
                  "checks": {name: True for name in CHECKS},
                  "total_seconds": (time.perf_counter_ns() - started) / 1e9,
                  "scope": "Actual CPU RGBD bytes and finite FP32 reconstruction. FP64 rounds to FP32. Floating arithmetic interoperation. No compression or integer-only training claim."}
        validate_rgbd_evidence(result, ROOT, require_release=args.samples >= 1_000_000)
        write_json(immutable_result, result)
        write_json(args.output, result)
        return result
    except Exception as error:
        failure = {"schema_version": 1, "status": "failed", "run_id": run_id, "config": config,
                   "provenance": provenance, "error_type": type(error).__name__, "error": str(error)}
        write_json(immutable_result, failure)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7331)
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "rgbd-token-verification.json")
    parser.add_argument("--child-timeout", type=int, default=120)
    parser.add_argument("--decode-file", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args(argv)
    if args.decode_file is not None:
        if args.expected_sha256 is None:
            parser.error("--decode-file requires --expected-sha256")
        receipt = decode_saved_file(args.decode_file, args.expected_sha256)
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if args.expected_sha256 is not None:
        parser.error("--expected-sha256 requires --decode-file")
    result = verify(args)
    print(json.dumps({"status": result["status"], "run_id": result["run_id"],
                      "samples": result["config"]["sample_count"], "checks": result["checks"],
                      "output": str(args.output.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
