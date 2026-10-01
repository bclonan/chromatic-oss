"""Verify the native hue library, actual wheels, complete tests and color arithmetic."""
from __future__ import annotations

from datetime import datetime, timezone
from email.parser import Parser
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from collections import Counter
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.provenance import capture_provenance, verify_provenance, file_hash, unique_run_id
from experiments.verification import verify_test_report

PACKAGE_CHECKS = {"isolated_package", "no_tensor_runtime", "native_color_encode_decode",
                  "actual_D_refinement", "native_matmul", "native_shapes", "native_file_roundtrip"}


def validate_package_row(row, actual, wheel_index):
    """Require executed checks, interpreter identity and exact installed package bytes."""
    label = row["environment"]
    minor = "3." + str(int(label[-2:]))
    expected_platform = "windows-x86_64" if label.startswith("python") else "linux-x86_64"
    binary_hash, binding_hash, platform_name = wheel_index[row["wheel_sha256"]]
    aggregate_keys = {"environment", "wheel_sha256", "receipt", "image_id", "network_disabled"}
    if (row["dependencies"] != [] or row["isolated"] is not True
            or set(actual) != set(row) - aggregate_keys
            or set(row["checks"]) != PACKAGE_CHECKS
            or any(value is not True for value in row["checks"].values())
            or not row["python"].startswith(minor + ".")
            or platform_name != expected_platform
            or row["native"]["sha256"] != binary_hash
            or row["module_sha256"] != binding_hash
            or row["native"]["abi_version"] != 1 or row["native"]["phase_modulus"] != 391680
            or any(actual[key] != row[key] for key in actual)
            or label.startswith("linux") and row.get("network_disabled") is not True):
        raise ValueError("Installed native wheel receipt disagrees with interpreter or package bytes")


def validate_full_collection(evidence, collection, xml):
    command = evidence["config"]["command"]
    relative_xml = evidence["config"].get("xml", "")
    if (not isinstance(relative_xml, str) or not relative_xml or Path(relative_xml).is_absolute()
            or ".." in Path(relative_xml).parts or not Path(xml).as_posix().endswith("/" + relative_xml)
            or command[0] != "<interpreter>"
            or command[1:] != ["-m", "pytest", "tests", "-q", "--junitxml=" + relative_xml]
            or collection["command"] != [command[0], "-m", "pytest", "tests", "--collect-only", "-q"]):
        raise ValueError("Full test receipt used a filtered or different runner")
    expected = []
    for nodeid in collection["nodeids"]:
        parts = nodeid.split("::")
        classname = parts[0][:-3].replace("/", ".")
        if len(parts) > 2:
            classname += "." + ".".join(parts[1:-1])
        expected.append((classname, parts[-1]))
    actual = [(row.attrib.get("classname"), row.attrib.get("name")) for row in ET.parse(xml).getroot().iter("testcase")]
    if not expected or len(set(expected)) != len(expected) or Counter(expected) != Counter(actual):
        raise ValueError("Executed tests differ from complete current-source collection")


def validate_native_commands(evidence, root):
    from scripts.build_native_hue import RUST_IMAGE, rust_notices, validate_toolchain
    commands = evidence["commands"]
    notices, _ = rust_notices(root)
    validate_toolchain(evidence["config"].get("rustc", ""), notices)
    if not commands or not commands[0]["stdout"].splitlines():
        raise ValueError("Native build did not record its Rust compiler version")
    validate_toolchain(commands[0]["stdout"].splitlines()[0], notices)
    if evidence["config"]["platform"] == "host":
        expected = [["rustc", "--version"]] + [["cargo", action, *( ["--release"] if action == "build" else [] ),
                     "--offline", "--manifest-path", str(Path("native/hue_tensor/Cargo.toml"))]
                    for action in ("test", "build")]
        if [row["argv"] for row in commands] != expected:
            raise ValueError("Native Windows build did not execute the required offline Rust commands")
    elif evidence["config"]["platform"] == "linux":
        args = commands[0]["argv"]
        shell = ("rustc --version && cargo test --offline --manifest-path /workspace/native/hue_tensor/Cargo.toml && "
                 "cargo build --release --offline --manifest-path /workspace/native/hue_tensor/Cargo.toml")
        if (len(commands) != 1 or args[:6] != ["docker", "run", "--rm", "--network", "none", "--mount"]
                or args[-4:] != [RUST_IMAGE, "sh", "-c", shell]
                or evidence["config"]["rust_image"] != RUST_IMAGE):
            raise ValueError("Native Linux build did not execute the pinned offline Rust image")
    else:
        raise ValueError("Native build platform is unsupported")


def validate_cli_demo(result, saved, root=ROOT):
    """Replay CLI contraction from its inputs, then load its actual output file."""
    from chromatensor import HueTensor
    modulus = 391680
    left, right = result["left"]["phases"], result["right"]["phases"]
    reference = lambda values: [sum(values[r * 2 + k] * right[k * 2 + c] for k in range(2)) % modulus
                                for r in range(2) for c in range(2)]
    expected = reference(left)
    mutation, persistence = result["mutation"], result["persistence"]
    changed = list(left)
    changed[0] = (changed[0] + 1) % modulus
    changed_expected = reference(changed)
    if (result.get("format") != "hue-native-demo-v1" or result["modulus"] != modulus
            or result["matmul"]["phases"] != expected or result["reference_phases"] != expected
            or mutation["channel"] != "D" or mutation["before"][:3] != mutation["after"][:3]
            or mutation["after"][3] != (mutation["before"][3] + 1) % 256
            or mutation["after_phase"] != mutation["before_phase"] + 1
            or mutation["changed_matmul"]["phases"] != changed_expected
            or mutation["reference_phases"] != changed_expected or changed_expected == expected
            or persistence["retained"] is not True or persistence["exact_roundtrip"] is not True
            or Path(persistence["path"]).is_absolute()
            or (Path(root) / persistence["path"]).resolve() != saved.resolve()
            or persistence["payload_bytes"] != 16 or persistence["file_bytes"] != saved.stat().st_size):
        raise ValueError("Standalone native CLI demonstration differs from independent arithmetic")
    with HueTensor.load(saved) as tensor:
        if tensor.shape != (2, 2) or list(tensor.phases()) != changed_expected:
            raise ValueError("Standalone native CLI saved tensor differs from its oracle")


def verify(root=ROOT, output_path=None):
    root = Path(root).resolve()
    config = {"profile": "native_hue", "tests": "results/test-verification.json",
              "xml": "artifacts/verification-tests.xml", "proof": "results/native-hue-verification.json",
              "packages": "results/native-hue-package-verification.json",
              "builds": ["results/native-hue-build-windows-x86_64.json", "results/native-hue-build-linux-x86_64.json"]}
    gates, artifacts = [], []

    def read(path):
        location = root / path
        value = json.loads(location.read_text(encoding="utf-8"))
        verify_provenance(value["provenance"], value["config"], root)
        artifacts.append(location)
        return value

    def gate(name, check):
        try:
            evidence = check()
            gates.append({"gate": name, "verified": True, "evidence": evidence})
        except Exception as error:
            gates.append({"gate": name, "verified": False, "evidence": str(error)})

    def host_tests():
        evidence = read(config["tests"])
        artifacts.append(root / config["xml"])
        result = verify_test_report(root / config["xml"], root, evidence)
        collection_path = root / evidence["collection"]
        verify_provenance(evidence["provenance"], evidence["config"], root, required_artifacts=[collection_path])
        validate_full_collection(evidence, json.loads(collection_path.read_text(encoding="utf-8")), root / config["xml"])
        artifacts.append(collection_path)
        return result

    def native_build(path):
        evidence = read(path)
        if evidence["config"].get("test") is not True or not evidence["commands"]:
            raise ValueError("Native build lacks executed Rust tests")
        validate_native_commands(evidence, root)
        output = "\n".join(row["stdout"] + row["stderr"] for row in evidence["commands"])
        results = [tuple(map(int, row)) for row in re.findall(r"test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored", output)]
        if (not all(row["returncode"] == 0 for row in evidence["commands"])
                or len(results) < 2 or results[0][0] < 22 or results[1][0] < 4
                or any(row[1] or row[2] for row in results)):
            raise ValueError("Native Rust tests failed, skipped or were reduced")
        for key in ("binary", "cli", "wheel"):
            item = evidence[key]
            if file_hash(root / item["path"]) != item["sha256"] or (root / item["path"]).stat().st_size != item["bytes"]:
                raise ValueError("Native build artifact changed")
        verify_provenance(evidence["provenance"], evidence["config"], root,
                          required_artifacts=[evidence[key]["path"] for key in ("binary", "cli", "wheel")])
        return {"rust_tests": sum(row[0] for row in results), "binary": evidence["binary"], "wheel": evidence["wheel"]}

    def wheels():
        evidence = read(config["packages"])
        expected = {prefix + version for prefix in ("python", "linux") for version in ("311", "312", "313", "314")}
        rows = evidence["environments"]
        if len(rows) != 8 or {row["environment"] for row in rows} != expected or evidence["config"]["runtime_dependencies"]:
            raise ValueError("Eight dependency-free native package environments are required")
        verify_provenance(evidence["provenance"], evidence["config"], root,
                          required_artifacts=[item["path"] for item in evidence["wheels"]] + [row["receipt"] for row in rows])
        wheel_index = {}
        for item in evidence["wheels"]:
            path = root / item["path"]
            if file_hash(path) != item["sha256"]:
                raise ValueError("Native wheel hash changed")
            with zipfile.ZipFile(path) as archive:
                metadata = Parser().parsestr(archive.read("chromatic_hue-0.1.0.dist-info/METADATA").decode("utf-8"))
                from scripts.build_native_hue import rust_notices
                _, notices = rust_notices(root)
                notice_names = {"rust/" + path.relative_to(root / "native/rust-notices").as_posix() for path in notices}
                platform_tag = "win_amd64" if path.name.endswith("win_amd64.whl") else "linux_x86_64"
                wheel_metadata = Parser().parsestr(archive.read("chromatic_hue-0.1.0.dist-info/WHEEL").decode("utf-8"))
                if (metadata.get("Name") != "chromatic-hue" or metadata.get("Version") != "0.1.0"
                        or metadata.get("Requires-Python") != ">=3.11" or metadata.get_all("Requires-Dist", [])
                        or metadata.get("License-Expression") != "AGPL-3.0-only"
                        or metadata.get("Author") != "Brad Clonan"
                        or set(metadata.get_all("License-File", [])) != {"LICENSE", "THIRD_PARTY_NOTICES.md", "COPYRIGHT"} | notice_names
                        or wheel_metadata.get("Tag") != "py3-none-" + platform_tag
                        or wheel_metadata.get("Root-Is-Purelib") != "false"):
                    raise ValueError("Native wheel metadata has unsupported identity, dependencies or platform")
                for license_name in ("LICENSE", "THIRD_PARTY_NOTICES.md", "COPYRIGHT"):
                    member = "chromatic_hue-0.1.0.dist-info/licenses/" + license_name
                    if archive.read(member) != (root / license_name).read_bytes():
                        raise ValueError("Packaged license or attribution differs from release source")
                for path in notices:
                    member = "chromatic_hue-0.1.0.dist-info/licenses/rust/" + path.relative_to(root / "native/rust-notices").as_posix()
                    if archive.read(member) != path.read_bytes():
                        raise ValueError("Packaged Rust attribution differs from reviewed toolchain notices")
                binding = archive.read("chromatic_hue/__init__.py")
                if binding != (root / "chromatensor/hue.py").read_bytes():
                    raise ValueError("Packaged binding differs from current source")
                binaries = [name for name in archive.namelist() if name.endswith((".dll", ".so"))]
                if len(binaries) != 1:
                    raise ValueError("Each native platform wheel needs exactly one library")
                binary_hash = hashlib.sha256(archive.read(binaries[0])).hexdigest()
                for source in (root / "native/hue_tensor").rglob("*"):
                    if source.is_file() and "target" not in source.parts and source.suffix in {".rs", ".toml", ".lock", ".h"}:
                        member = "chromatic_hue/_source/" + source.relative_to(root / "native/hue_tensor").as_posix()
                        if archive.read(member) != source.read_bytes():
                            raise ValueError("Packaged Rust source differs from current source")
                platform_name = binaries[0].split("/")[-2]
                build_path = config["builds"][0 if platform_name == "windows-x86_64" else 1]
                build = read(build_path)
                if build["binary"]["sha256"] != binary_hash or build["wheel"] != item:
                    raise ValueError("Installed wheel differs from its tested native build")
                wheel_index[item["sha256"]] = (binary_hash, hashlib.sha256(binding).hexdigest(), platform_name)
        if len(wheel_index) != 2:
            raise ValueError("Two distinct native platform wheels are required")
        for row in rows:
            actual = json.loads((root / row["receipt"]).read_text(encoding="utf-8"))
            validate_package_row(row, actual, wheel_index)
        return {"clean_environments": 8, "runtime_dependencies": [], "wheels": evidence["wheels"]}

    def independent_proof():
        from scripts.verify_native_hue import validate_native_hue_receipt
        result = read(config["proof"])
        validate_native_hue_receipt(result, root, require_million=True)
        return {"run_id": result["run_id"], "samples": result["config"]["samples"],
                "operations": sorted(result["operations"]), "payload_bytes": result["tensor_file"]["payload_bytes"],
                "file_bytes": result["tensor_file"]["bytes"], "cold_process": result["cold_process"]["receipt"]}

    def cli():
        build = read(config["builds"][0])
        binary = root / build["cli"]["path"]
        directory = root / "artifacts/native/acceptance-cli"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / (unique_run_id("cli") + ".hue")
        process = subprocess.run([str(binary), "demo", destination.relative_to(root).as_posix()], cwd=root,
                                 check=True, capture_output=True, text=True, timeout=30)
        result = json.loads(process.stdout)
        validate_cli_demo(result, destination, root)
        receipt = destination.with_suffix(".json")
        receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        artifacts.extend([binary, destination, receipt])
        return {"binary_sha256": file_hash(binary), "receipt": receipt.relative_to(root).as_posix(), "demo": result}

    gate("Complete current-source Python tests including native hue behavior", host_tests)
    for path in config["builds"]:
        gate("Native Rust build and executed tests on " + ("Windows" if "windows" in path else "Linux"), lambda path=path: native_build(path))
    gate("Eight clean installed native hue platform wheels without ML dependencies", wheels)
    gate("Million-value independent hue oracle, matrix operations and file-only decoding", independent_proof)
    gate("Standalone native CLI color encoding, decoding and contraction", cli)
    report = {"schema_version": 3, "profile": "native_hue", "verified_at": datetime.now(timezone.utc).isoformat(),
              "config": config, "gates": gates, "complete": all(row["verified"] for row in gates)}
    report["provenance"] = capture_provenance(config, root, artifacts)
    path = Path(output_path or root / "results/acceptance.json")
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(verify())
