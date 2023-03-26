import pytest
from transformer_lens.HookedTransformer import MaskedHookedTransformer
import torch


@pytest.fixture
def gpt2():
    return MaskedHookedTransformer.from_pretrained("gpt2")


def test_regularizer(gpt2):
    from train import regularizer

    regularizer_loss = regularizer(gpt2, beta=0)
    assert regularizer_loss == 1 / (1 + torch.exp(-1))
