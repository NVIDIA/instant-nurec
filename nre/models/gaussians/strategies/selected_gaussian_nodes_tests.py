from typing import cast

import torch

from nre.models.gaussians.gaussians_model import BaseGaussianModel
from nre.models.gaussians.strategies.selected_gaussian_nodes import SelectedGaussianNodes
from nre.models.nn_extensions import TypedModuleDict


class DummyGaussian(BaseGaussianModel):
    def __init__(self, num_gaussians, densities, scales):
        self._num = num_gaussians
        self._densities = densities
        self._scales = scales
        self.optimizers = []  # no optimizers for this dummy

    def get_num_gaussians(self):
        return self._num

    def get_densities(self, preactivation=False):
        return self._densities

    def get_scales(self, preactivation=False):
        return self._scales

    def density_activation_inv(self, x):
        return x

    def scale_activation_inv(self, x):
        return x


# Helper to stub optimizer state buffers
def make_opt(buff_shape):
    class OptStub:
        def state_dict(self):
            return {"param_groups": [{"name": "param", "params": [0]}], "state": {0: {"buf": torch.zeros(buff_shape)}}}

    return OptStub()


# DummyGaussian subclass that injects our stub optimizers
class DummyGaussianWithBuffer(DummyGaussian):
    def __init__(self, buff_shape):
        super().__init__(3, torch.zeros(3), torch.zeros(3))
        # one optimizer entry with stub buffer
        self.optimizers = [{"optimizer": make_opt(buff_shape)}]


def test_zero_layers():
    nodes = cast(
        TypedModuleDict[BaseGaussianModel],
        TypedModuleDict({"layer0": DummyGaussian(0, torch.tensor([]), torch.tensor([]))}),
    )
    sel = SelectedGaussianNodes(nodes, exclude_layer_ids=[])
    # No layers should be included
    assert sel.layer_ids == []
    # Densities and scales should be empty
    dens = sel.get_densities()
    scales = sel.get_scales()
    assert dens.numel() == 0
    assert scales.numel() == 0
    # Optimizer states should be empty dict
    assert sel.get_optimizer_states() == {}


def test_mixed_layers():
    d0 = DummyGaussian(0, torch.tensor([]), torch.tensor([]))
    d1 = DummyGaussian(2, torch.tensor([1.0, 2.0]), torch.tensor([0.5, 0.6]))
    nodes = cast(TypedModuleDict[BaseGaussianModel], TypedModuleDict({"zero": d0, "nonzero": d1}))
    sel = SelectedGaussianNodes(nodes, exclude_layer_ids=[])
    # Only 'nonzero' should be present
    assert sel.layer_ids == ["nonzero"]
    assert torch.allclose(sel.get_densities(), torch.tensor([1.0, 2.0]))
    assert torch.allclose(sel.get_scales(), torch.tensor([0.5, 0.6]))


def test_optimizer_state_zero_element_reshape():
    # Cast to satisfy TypedModuleDict signature using our buffer-capable Dummy
    nodes = cast(
        TypedModuleDict[BaseGaussianModel],
        TypedModuleDict(
            {
                "l1": DummyGaussianWithBuffer((0, 5, 3)),
                "l2": DummyGaussianWithBuffer((0, 10, 3)),
            }
        ),
    )
    sel = SelectedGaussianNodes(nodes, exclude_layer_ids=[])
    states = sel.get_optimizer_states()
    # The buffer should appear under parameter name 'param' and key 'buf'
    assert "param" in states
    assert "buf" in states["param"]
    buf_tensor = states["param"]["buf"]
    # Should succeed and yield shape (0, 10, 3)
    assert buf_tensor.shape == (0, 10, 3)
