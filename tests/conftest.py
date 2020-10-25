import pytest
import torch
import geoopt


@pytest.fixture(autouse=True, params=[42, 41])
def seed(request):
    torch.manual_seed(request.param)


@pytest.fixture(params=[-1, 0, 1])
def curvature_1(request):
    return request.param


@pytest.fixture(params=[-1, 0, 1])
def curvature_2(request):
    return request.param


@pytest.fixture()
def ball_1(curvature_1):
    return geoopt.Stereographic(curvature_1)


@pytest.fixture()
def ball_2(curvature_2):
    return geoopt.Stereographic(curvature_2)


@pytest.fixture(params=[-1, 0, 1])
def curvature(request):
    return request.param


@pytest.fixture()
def ball(curvature):
    return geoopt.Stereographic(curvature)
