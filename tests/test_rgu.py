import pytest
from types import SimpleNamespace
from iguane.fom import fom_ugr, RAWDATA
import json


@pytest.mark.parametrize("fom_version,norm", [
    pytest.param("1.0", False, id="1.0"),
    pytest.param("1.0", True,  id="1.0-renorm"),
])
def test_rgu_v1(fom_version, norm, file_regression):
    args = SimpleNamespace(fom_version=fom_version, custom_weights=None, norm=norm)
    gpus = sorted(RAWDATA.keys())
    rgu = {gpu: fom_ugr(gpu, args=args) for gpu in gpus}
    label = f"{fom_version}-renorm" if norm else fom_version
    file_regression.check(f"RGU {label}:\n\n{json.dumps(rgu, indent=1)}\n")
