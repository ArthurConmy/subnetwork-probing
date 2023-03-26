from functools import partial
from typing import Dict, List

import IPython
import numpy as np
import plotly
import torch
import torch.nn.functional as F
import transformer_lens.utils as utils
import wandb
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.ioi_dataset import IOIDataset

from classifiers import NERModel, POSModel, UDModel
from subnetwork_datasets import (build_vocab, evaluate, load_conllu, load_ner,
                                 masked_loss, sent_avgs)
from util import from_numpy, partial_state_dict

N = 100


def log_plotly_bar_chart(x: List[str], y: List[float]) -> None:
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    wandb.log({"mask_scores": fig})


def visualize_mask(gpt2: HookedTransformer) -> None:
    node_name = []
    mask_scores_for_names = []
    total_nodes = 0
    nodes_to_mask = []
    for layer_index, layer in enumerate(gpt2.blocks):
        for head in range(12):
            for q_k_v in ["q", "k", "v"]:
                total_nodes += 1
                if q_k_v == "q":

                    if layer.attn.hook_q.sample_mask()[layer_index].cpu().item() < 0.5:
                        nodes_to_mask.append(f"{layer_index}.{head}.{q_k_v}")
                if q_k_v == "k":
                    if layer.attn.hook_k.sample_mask()[layer_index].cpu().item() < 0.5:
                        nodes_to_mask.append(f"{layer_index}.{head}.{q_k_v}")
                if q_k_v == "v":
                    if layer.attn.hook_v.sample_mask()[layer_index].cpu().item() < 0.5:
                        nodes_to_mask.append(f"{layer_index}.{head}.{q_k_v}")

    log_plotly_bar_chart(x=node_name, y=mask_scores_for_names)
    node_count = total_nodes - len(nodes_to_mask)
    return node_count, nodes_to_mask


def regularizer(
    gpt2: HookedTransformer,
    gamma: float = -0.1,
    zeta: float = 1.1,
    beta: float = 2 / 3,
) -> torch.Tensor:
    # TODO: globally read hyperparams from config
    # need to also do this in the masked hook point so
    # the hyperparams are the same
    def regularization_term(mask: torch.nn.Parameter) -> torch.Tensor:
        return torch.sigmoid(mask - beta * np.log(-gamma / zeta)).mean()

    mask_scores = [
        regularization_term(p)
        for (n, p) in gpt2.named_parameters()
        if "mask_scores" in n
    ]
    return torch.mean(torch.stack(mask_scores))


def logit_diff_from_ioi_dataset(
    logits: torch.Tensor, tokens: torch.Tensor, mean=False,
):
    assert tokens.shape == (
        N,
        16,
    ), tokens.shape  # TODO check this is not breaking things...
    assert len(logits.shape) == 3, logits.shape

    io_labels = tokens[:, 2]
    s_labels = tokens[:, 4]

    io_logits = logits[torch.arange(N), -2, io_labels]
    s_logits = logits[torch.arange(N), -2, s_labels]

    logit_diff = io_logits - s_logits
    if mean:
        return logit_diff.mean()
    else:
        return logit_diff


def do_random_resample_caching(
    gpt2: HookedTransformer, train_data: torch.Tensor
) -> HookedTransformer:
    for layer in gpt2.blocks:
        layer.attn.hook_q.is_caching = True
        layer.attn.hook_k.is_caching = True
        layer.attn.hook_v.is_caching = True

    _ = gpt2(train_data)

    for layer in gpt2.blocks:
        layer.attn.hook_q.is_caching = False
        layer.attn.hook_k.is_caching = False
        layer.attn.hook_v.is_caching = False

    return gpt2


def train_ioi(
    gpt2, mask_lr=0.01, epochs=2000, verbose=True, lambda_reg=100,
):
    wandb.init(
        project="subnetwork-probing",
        entity="acdcremix",
        config={"epochs": epochs, "mask_lr": mask_lr, "lambda_reg": lambda_reg},
    )
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
    log = []
    from tqdm import tqdm

    for epoch in tqdm(range(epochs)):  # tqdm.notebook.tqdm(range(epochs)):
        gpt2 = do_random_resample_caching(gpt2, train_data)
        gpt2.train()
        trainer.zero_grad()
        # compute loss, also log other metrics
        logit_diff_term = -1.0 * logit_diff_from_ioi_dataset(
            gpt2(train_data), train_data, mean=True
        )
        regularizer_term = regularizer(gpt2)
        loss = logit_diff_term + lambda_reg * regularizer_term
        loss.backward()

        wandb.log(
            {
                "regularisation_loss": regularizer_term,
                "logit_diff_loss": logit_diff_term,
                "total_loss": loss,
            }
        )
        trainer.step()

        log.append({"loss_val": loss.item()})
        if epoch % 10 == 0:
            number_of_nodes, nodes_to_mask = visualize_mask(gpt2)
    # wandb.finish()
    # torch.save(gpt2.state_dict(), "masked_gpt2.pt")
    return log, gpt2, number_of_nodes, logit_diff_term, nodes_to_mask


def sanity_check_with_transformer_lens(mask_dict):
    ioi_dataset = IOIDataset(prompt_type="ABBA", N=N, nb_templates=1)
    train_data = ioi_dataset.toks.long()
    gpt2 = HookedTransformer.from_pretrained(is_masked=False, model_name="gpt2")
    gpt2.freeze_weights()
    logits = gpt2(train_data)
    logit_diff = logit_diff_from_ioi_dataset(logits, train_data, mean=True)

    fwd_hooks = make_forward_hooks(mask_dict)
    logits = gpt2.run_with_hooks(train_data, return_type="logits", fwd_hooks=fwd_hooks)
    logit_diff_masked = logit_diff_from_ioi_dataset(logits, train_data, mean=True)
    print("original logit diff", logit_diff)
    print("masked logit diff", logit_diff_masked)


def make_forward_hooks(mask_dict):
    forward_hooks = []
    for layer in range(12):
        for head in range(12):
            for qkv in ["q", "k", "v"]:
                mask_value = mask_dict[f"{layer}.{head}.{qkv}"]

                def head_ablation_hook(
                    value, hook, head_idx, layer_idx, qkv_val, mask_value
                ):
                    value[:, :, head_idx, :] *= mask_value
                    return value

                a_hook = (
                    utils.get_act_name(qkv, int(layer)),
                    partial(
                        head_ablation_hook,
                        head_idx=head,
                        layer_idx=layer,
                        qkv_val=qkv,
                        mask_value=mask_value,
                    ),
                )
                forward_hooks.append(a_hook)
    return forward_hooks


def log_percentage_binary(mask_val_dict: Dict) -> float:
    binary_count = 0
    total_count = 0
    for _, v in mask_val_dict.items():
        total_count += 1
        if v == 0 or v == 1:
            binary_count += 1
    return binary_count / total_count


def get_nodes_mask_dict(gpt2: HookedTransformer):
    mask_value_dict = {}
    for layer_index, layer in enumerate(gpt2.blocks):
        for head_index in range(12):
            for q_k_v in ["q", "k", "v"]:
                # total_nodes += 1
                if q_k_v == "q":
                    mask_value = (
                        layer.attn.hook_q.sample_mask()[head_index].cpu().item()
                    )
                if q_k_v == "k":
                    mask_value = (
                        layer.attn.hook_k.sample_mask()[head_index].cpu().item()
                    )
                if q_k_v == "v":
                    mask_value = (
                        layer.attn.hook_v.sample_mask()[head_index].cpu().item()
                    )
                mask_value_dict[f"{layer_index}.{head_index}.{q_k_v}"] = mask_value
    return mask_value_dict


if __name__ == "__main__":

    from transformer_lens.HookedTransformer import HookedTransformer

    regularization_params = [
        1e3,
        700,
        675,
        650,
        625,
        600,
        575,
        525,
        500,
        550,
        1e2,
        1e0,
        1e-1,
        1e-2,
    ]

    is_masked = True
    logit_diff_list = []
    number_of_nodes_list = []
    percentage_binary_list = []

    for a_regulation_param in regularization_params:
        for task in ["IOI"]:
            gpt2 = HookedTransformer.from_pretrained(
                is_masked=is_masked, model_name="gpt2"
            )
            gpt2.freeze_weights()
            print("Finding subnetwork...")
            assert task == "IOI"
            log, model, number_of_nodes, logit_diff, nodes_to_mask = train_ioi(
                gpt2, lambda_reg=a_regulation_param
            )
            print("nodes to mask", nodes_to_mask)
            logit_diff_list.append(logit_diff * -1)
            number_of_nodes_list.append(number_of_nodes)
            mask_val_dict = get_nodes_mask_dict(model)
            percentage_binary = log_percentage_binary(mask_val_dict)
            wandb.log({"percentage_binary": percentage_binary})
            percentage_binary_list.append(percentage_binary)
            sanity_check_with_transformer_lens(mask_val_dict)
            wandb.finish()

    wandb.init(project="pareto-subnetwork-probing", entity="acdcremix")
    import pandas as pd
    import plotly.express as px

    df = pd.DataFrame(
        {
            "x": number_of_nodes_list,
            "y": [i.cpu().detach().item() for i in logit_diff_list],
            "regularization_params": regularization_params,
            "percentage_binary": percentage_binary_list,
        }
    )
    plt = px.scatter(
        df, x="x", y="y", hover_data=["regularization_params", "percentage_binary"]
    )
    plt.update_layout(xaxis_title="Number of Nodes", yaxis_title="Logit Diff")
    wandb.log({"number_of_nodes": plt})
    wandb.finish()
