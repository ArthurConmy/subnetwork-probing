import wandb
import torch
from train import regularizer, logit_diff_from_ioi_dataset
from transformer_lens.ioi_dataset import IOIDataset
from transformer_lens.HookedTransformer import HookedTransformer

N = 100


def get_gradients(
    gpt2, mask_lr=0.01, epochs=2000, verbose=True, lambda_reg=100,
):
    # blackbox this bit
    ioi_dataset = IOIDataset(prompt_type="ABBA", N=N, nb_templates=1,)
    train_data = ioi_dataset.toks.long()

    # one parameter per thing that is masked
    mask_params = [
        p for n, p in gpt2.named_parameters() if "mask_scores" in n and p.requires_grad
    ]
    # parameters for the probe (we don't use a probe)
    gpt2_params = [
        p
        for n, p in gpt2.named_parameters()
        if "mask_scores" not in n and p.requires_grad
    ]
    assert len(gpt2_params) == 0, ("GPT2 should be empty", gpt2_params)
    trainer = torch.optim.Adam(mask_params, lr=mask_lr)

    gpt2.train()
    trainer.zero_grad()
    # compute loss, also log other metrics
    logit_diff_term = -1.0 * logit_diff_from_ioi_dataset(
        gpt2(train_data), train_data, mean=True
    )
    regularizer_term = regularizer(gpt2)
    loss = logit_diff_term + lambda_reg * regularizer_term
    loss.backward()

    return gpt2, train_data


def main():
    wandb.init(
        project="better_than_one", entity="acdcremix",
    )
    gpt2 = HookedTransformer.from_pretrained(is_masked=True, model_name="gpt2")
    gpt2.freeze_weights()
    gpt2, train_data = get_gradients(gpt2, lambda_reg=100)
    import ipdb

    ipdb.set_trace()


if __name__ == "__main__":
    main()
