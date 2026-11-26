import pytest
import torch

from hqprompt.losses import Sobel, StubDists, StubID, build_losses


@pytest.mark.parametrize("cls", [StubDists, StubID])
def test_stub_grad_and_identity(cls):
    m = cls()
    assert all(not p.requires_grad for p in m.parameters())
    x = torch.rand(2, 3, 64, 64)
    assert m(x, x).abs() < 1e-5
    p = torch.rand(2, 3, 64, 64, requires_grad=True)
    loss = m(p, x)
    assert loss.dtype == torch.float32 and loss.ndim == 0 and loss > 0
    loss.backward()
    assert p.grad is not None and p.grad.abs().sum() > 0


def test_sobel_range():
    y = Sobel()(torch.rand(3, 3, 32, 32))
    assert y.shape == (3, 3, 32, 32) and y.min() >= 0 and y.max() <= 1 + 1e-6


def test_build_losses_dry_run(hqp_root):
    d = build_losses(hqp_root, "cpu", dry_run=True)
    assert set(d) == {"ea_dists", "id"}
    assert isinstance(d["ea_dists"], StubDists) and isinstance(d["id"], StubID)


def test_real_ea_dists():
    pytest.importorskip("pyiqa")
    from hqprompt.losses import EADists
    m = EADists()
    x = torch.rand(1, 3, 64, 64, requires_grad=True)
    m(x, torch.rand(1, 3, 64, 64)).backward()
    assert x.grad is not None


def test_real_arcface(hqp_root):
    pytest.importorskip("facexlib")
    from hqprompt.losses import ArcFaceID
    m = ArcFaceID(hqp_root)
    x = torch.rand(1, 3, 128, 128, requires_grad=True)
    m(x, torch.rand(1, 3, 128, 128)).backward()
    assert x.grad is not None
