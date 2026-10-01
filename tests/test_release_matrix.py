import pytest

from scripts.verify_dependencies import ROOT, matrix_report


def report(tmp_path, special=""):
    cases = [f'<testcase classname="{path.stem}" name="test_contract"/>'
             for path in (ROOT / "tests").glob("test_*.py")]
    cases.append('<testcase classname="test_device" name="test_gpu">' + special + '</testcase>')
    path = tmp_path / "matrix.xml"
    path.write_text('<testsuites><testsuite>' + ''.join(cases) + '</testsuite></testsuites>')
    return path


def test_cpu_dependency_matrix_reports_cuda_limit(tmp_path):
    value = matrix_report(report(tmp_path, '<skipped message="CUDA device unavailable"/>'))
    assert value["cuda_only_skipped"] == 1
    assert value["passed"] == len(list((ROOT / "tests").glob("test_*.py")))


@pytest.mark.parametrize("special", ['<skipped message="training unavailable"/>', '<failure/>', '<error/>'])
def test_cpu_matrix_rejects_hidden_execution_gaps(tmp_path, special):
    with pytest.raises(ValueError):
        matrix_report(report(tmp_path, special))


def test_cpu_matrix_requires_complete_test_module_coverage(tmp_path):
    path = tmp_path / "partial.xml"
    path.write_text('<testsuites><testsuite><testcase classname="test_device" name="test_gpu"/></testsuite></testsuites>')
    with pytest.raises(ValueError, match="omitted"):
        matrix_report(path)
