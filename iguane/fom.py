import importlib.resources
import json


with importlib.resources.files("iguane").joinpath("gpuflops.json").open() as file:
    RAWDATA = json.load(file)


UGR_VERSIONS = {
    # v1.0 is equivalent to UGR
    # https://docs.alliancecan.ca/wiki/Allocations_and_compute_scheduling
    "1.0":        { 'fp16': 1.6, 'fp32': 1.6, 'memgb': 0.8 },
    "1.0-renorm": { 'fp16': 0.4, 'fp32': 0.4, 'memgb': 0.2 },
}


class FigureOfMerit(dict):
    def __call__(self, f):
        globals().setdefault('FOM', {})
        assert f.__name__.startswith('fom_') and f.__name__ != 'fom_', \
            'Figure-of-Merit function names must start with "fom_"!'
        self[f.__name__[4:]] = f
        return f


FOM = FigureOfMerit()


@FOM
def fom_count(name, *, args=None):
    return 1


@FOM
def fom_fp16(name, *, args=None):
    data = RAWDATA[name]
    return data['fp16'] or data['fp32']


@FOM
def fom_fp32(name, *, args=None):
    return RAWDATA[name]['fp32']


@FOM
def fom_fp64(name, *, args=None):
    return RAWDATA[name]['fp64']


@FOM
def fom_tf32(name, *, args=None):
    data = RAWDATA[name]
    return data['tf32'] or data['fp32']


@FOM
def fom_ugr(name, *, args=None):
    weights = UGR_VERSIONS[args.ugr_version if args else "1.0"]
    ref  = RAWDATA['A100-SXM4-40GB']
    data = RAWDATA[name].copy()
    data['tf32'] = data['tf32'] or data['fp32']
    data['fp16'] = data['fp16'] or data['fp32']
    return sum([w * (data[k] / ref[k]) for k, w in weights.items()])


@FOM
def fom_iguane(name, *, args=None):
    ref  = RAWDATA['A100-SXM4-80GB']
    data = RAWDATA[name].copy()
    data['tf32'] = data['tf32'] or data['fp32']
    data['fp16'] = data['fp16'] or data['fp32']
    weight_fp16  = 0.2
    weight_fp32  = 0.2
    weight_tf32  = 0.2
    weight_memgb = 0.2
    weight_membw = 0.2
    return weight_fp16  * (data['fp16']  / ref['fp16'])  + \
           weight_fp32  * (data['fp32']  / ref['fp32'])  + \
           weight_tf32  * (data['tf32']  / ref['tf32'])  + \
           weight_memgb * (data['memgb'] / ref['memgb']) + \
           weight_membw * (data['membw'] / ref['membw'])
