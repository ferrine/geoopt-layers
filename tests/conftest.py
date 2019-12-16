import pytest
import torch


@pytest.fixture(autouse=True, params=[42, 41])
def seed(request):
    torch.manual_seed(request.param)
