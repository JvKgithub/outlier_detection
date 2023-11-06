import torch

from src.losses_od.losses_od import NTXentLoss, OdLossTotal


# ---- Tests for NTXentLoss ----
def test_ntxentloss_basic_functionality():
    """Test whether correct type is returned."""
    loss_fn = NTXentLoss()
    embeddings = torch.randn((10, 5))
    loss = loss_fn(embeddings)
    assert isinstance(loss, torch.Tensor)


def test_ntxentloss_clip_too_high():
    """Test error behaviour when clipping is set too high."""
    # Temperature is set very low to ensure clipping is triggered, clipping is set so high that output will be nan
    loss_fn = NTXentLoss(in_temperature=1e-8, in_clip_max=1000)
    embeddings = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    loss = loss_fn(embeddings)
    assert torch.any(torch.isnan(loss))


def test_ntxentloss_clipping_correct():
    """Test clipping behaviour when default clipping value is used. Output should not be nan"""
    # Temperature is set very low to ensure clipping is triggered, clipping is set to default value, output should not be nan
    loss_fn = NTXentLoss(in_temperature=1e-8)
    embeddings = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    loss = loss_fn(embeddings)
    assert not torch.any(torch.isnan(loss))


# ---- Tests for OdLossTotal ----
def test_odlosstotal_basic_functionality():
    """Ensure basic functionality and outputs for the OdLossTotal loss function."""
    loss_weights = {
        'weight_sim': 1.0,
        'weight_shift': 1.0,
        'weight_cls': 1.0,
        'weight_norm': 1.0
    }
    loss_fn = OdLossTotal(2, loss_weights)
    predictions = {
        'sim': torch.randn((10, 5)),
        'shift': torch.randn((10, 3)),
        'cls': torch.randn((10, 3))
    }
    labels_shift = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    losses = loss_fn(predictions, labels_shift)
    assert isinstance(losses, dict)
    assert 'loss_total' in losses


def test_odlosstotal_weights_behavior():
    """Test that only specified loss components are activated."""

    # Only activating the NT-Xent similarity loss
    loss_weights = {
        'weight_sim': 1.0,
        'weight_shift': 0.0,
        'weight_cls': 0.0,
        'weight_norm': 0.0
    }
    loss_fn = OdLossTotal(2, loss_weights)
    predictions = {
        'sim': torch.randn((10, 5)),
        'shift': torch.randn((10, 3)),
        'cls': torch.randn((10, 3))
    }
    labels_shift = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    losses = loss_fn(predictions, labels_shift)
    assert 'loss_sim' in losses
    assert 'loss_shift' not in losses
    assert 'loss_cls' not in losses
    assert 'loss_norm' not in losses


def test_odlosstotal_loss_combination():
    """Verify correct aggregation of all loss components to the total loss."""
    loss_weights = {
        'weight_sim': 1.0,
        'weight_shift': 1.0,
        'weight_cls': 1.0,
        'weight_norm': 1.0
    }
    loss_fn = OdLossTotal(2, loss_weights)
    predictions = {
        'sim': torch.randn((10, 5)),
        'shift': torch.randn((10, 3)),
        'cls': torch.randn((10, 3))
    }
    labels_shift = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    losses = loss_fn(predictions, labels_shift)
    total = (losses.get('loss_sim', 0) +
             losses.get('loss_shift', 0) +
             losses.get('loss_cls', 0) +
             losses.get('loss_norm', 0))
    assert torch.isclose(losses['loss_total'], total, atol=1e-5)


def test_odlosstotal_lrfinder_behavior():
    """Check if the behavior aligns with the learning rate finder mode. Where only one value and not a dict should be returned."""
    loss_weights = {
        'weight_sim': 1.0,
        'weight_shift': 1.0,
        'weight_cls': 1.0,
        'weight_norm': 1.0
    }
    loss_fn = OdLossTotal(2, loss_weights, in_lr_finder=True)
    predictions = {
        'sim': torch.randn((10, 5)),
        'shift': torch.randn((10, 3)),
        'cls': torch.randn((10, 3))
    }
    labels_shift = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    loss = loss_fn(predictions, labels_shift)
    assert isinstance(loss, torch.Tensor)
