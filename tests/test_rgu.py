from types import SimpleNamespace
import pytest
from iguane.fom import fom_ugr, RAWDATA
import json


@pytest.mark.parametrize("rgu_version", ["1.0", "1.0-renorm"])
def test_rgu_v1(rgu_version, file_regression):
    args = SimpleNamespace(ugr_version=rgu_version)
    gpus = sorted(RAWDATA.keys())
    rgu = {gpu: fom_ugr(gpu, args=args) for gpu in gpus}
    file_regression.check(f"RGU {rgu_version}:\n\n{json.dumps(rgu, indent=1)}\n")
