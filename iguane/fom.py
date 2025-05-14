import copy
import importlib.resources

try:
    import tomllib
except ModuleNotFoundError:
    # for Python before 3.11
    import pip._vendor.tomli as tomllib

from iguane.log import logger


# Import RAWDATA (a dictionary mapping GPU name to GPU flops and data) from a resource file
RAWDATA = tomllib.loads(importlib.resources.read_text('iguane', 'rawdata.toml'))
FIELDS = [
    k for k, v in next(iter(RAWDATA.values())).items()
    # bool is redundant and is interpreted as an int but it is clearer this way
    if isinstance(v, (bool, int, float))
]


#
# FoM versions
#
#   v1.0 is equivalent to RGU/UGR:
#   https://docs.alliancecan.ca/wiki/Allocations_and_compute_scheduling
#
# fmt: off
FOM_VERSIONS = {
    '1.0':      { 'ref': 'A100-SXM4-40GB', 'fp16': 1.6, 'fp32': 1.6,              'memgb': 0.8                 },
    '2.0-0':    { 'ref': 'A100-SXM4-80GB', 'fp16': 0.2, 'fp32': 0.1, 'tf32': 0.2, 'memgb': 0.25, 'membw': 0.25 },
    '2.0':      { 'ref': 'A100-SXM4-80GB', 'fp16': 0.2, 'fp32': 0.1, 'tf32': 0.2, 'memgb': 0.25, 'membw': 0.25 },
    'iguane-0': { 'ref': 'A100-SXM4-80GB', 'fp16': 0.2, 'fp32': 0.2, 'tf32': 0.2, 'memgb': 0.2,  'membw': 0.2  },
}
# fmt: on
_CURRENT_FOM_VERSION = '1.0'
FOM_VERSIONS['ugr'] = FOM_VERSIONS[_CURRENT_FOM_VERSION]


#
# A decorator that automatically registers the FoM implementation function's
# name in a dictionary iguane.FOM, for use by the arguments-parsing logic.
#
def fom(f):
    globals().setdefault('FOM', {})
    assert f.__name__.startswith('fom_') and f.__name__ != 'fom_', \
           'Figure-of-Merit function names must start with "fom_"!'
    FOM[f.__name__[4:]] = f
    return f


@fom
def fom_count(gpu_name, *, args=None):
    return 1


@fom
def fom_fp16(gpu_name, *, args=None):
    data = RAWDATA[gpu_name]
    return data['fp16'] or data['fp32']


@fom
def fom_fp32(gpu_name, *, args=None):
    return RAWDATA[gpu_name]['fp32']


@fom
def fom_fp64(gpu_name, *, args=None):
    return RAWDATA[gpu_name]['fp64']


@fom
def fom_tf32(gpu_name, *, args=None):
    data = RAWDATA[gpu_name]
    return data['tf32'] or data['fp32']


@fom
def fom_ugr(gpu_name, *, args=None):
    fom_version  = args and args.fom_version    or _CURRENT_FOM_VERSION
    weights      = args and args.custom_weights or FOM_VERSIONS[fom_version]
    weights      = weights.copy()
    gpu_name_ref = weights.pop('ref', 'A100-SXM4-40GB')
    weights_norm = args and args.norm and sum(weights.values()) or 1.0

    ref          = RAWDATA[gpu_name_ref].copy()
    data         = RAWDATA[gpu_name].copy()
    data['tf32'] = data['tf32'] or data['fp32']
    data['fp16'] = data['fp16'] or data['fp32']

    _sum = 0
    for attr, weight in weights.items():
        ratio = (weight / weights_norm) * (data[attr] / ref[attr])
        _sum += ratio
        logger.debug(f"{attr}\t{ratio:.4f}\t{gpu_name}")

    return _sum
