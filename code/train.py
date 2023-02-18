import IPython
import torch
import torch.nn.functional as F
import numpy as np
from transformer_lens.HookedTransformer import HookedTransformer
from util import from_numpy, partial_state_dict
from classifiers import POSModel, NERModel, UDModel
from subnetwork_datasets import (
    load_conllu,
    build_vocab,
    sent_avgs,
    masked_loss,
    evaluate,
    load_ner,
)
from transformer_lens.ioi_dataset import IOIDataset
import wandb
import plotly
from typing import List
import transformer_lens.utils as utils

N = 100


def log_plotly_bar_chart(x: List[str], y: List[float]) -> None:
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    wandb.log({"mask_scores": fig})


def visualize_mask(gpt2: HookedTransformer) -> None:
    node_name = []
    mask_scores_for_names = []
    node_count = 0
    nodes_to_mask = []
    for name, param in gpt2.named_parameters():
        if "mask_scores" in name:
            if "attn" not in name:  # TODO: MLPs mask scores
                import ipdb

                ipdb.set_trace()
            for head_index, mask_value in enumerate(param[:, 0]):
                mask_scores_for_names.append(mask_value.detach().cpu().item())
                layer = name.split(".")[1]
                qkv = name.split(".")[3].split("_")[1]
                node_name.append(f"{layer}.{head_index}.{qkv}")
                if mask_scores_for_names[-1] > 0.0:
                    node_count += 1
                elif mask_scores_for_names[-1] < 0.0:
                    nodes_to_mask.append(node_name[-1])
    log_plotly_bar_chart(x=node_name, y=mask_scores_for_names)
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


def train_ioi(
    gpt2, mask_lr=0.01, epochs=1000, verbose=True, lambda_reg=100,
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
    wandb.finish()
    return log, gpt2, number_of_nodes, logit_diff_term, nodes_to_mask


def sanity_check_with_transformer_lens(nodes_to_mask):
    import ipdb

    ipdb.set_trace()
    ioi_dataset = IOIDataset(prompt_type="ABBA", N=N, nb_templates=1,)
    train_data = ioi_dataset.toks.long()
    gpt2 = HookedTransformer.from_pretrained(is_masked=False, model_name="gpt2")
    gpt2.freeze_weights()
    logits = gpt2(train_data)
    logit_diff = logit_diff_from_ioi_dataset(logits, train_data, mean=True)

    fwd_hooks = make_forward_hooks(nodes_to_mask)
    logits = gpt2.run_with_hooks(train_data, return_type="logits", fwd_hooks=fwd_hooks)
    logit_diff_masked = logit_diff_from_ioi_dataset(logits, train_data, mean=True)
    print("original logit diff", logit_diff)
    print("masked logit diff", logit_diff_masked)


def make_forward_hooks(nodes_to_mask):
    forward_hooks = []
    for node in nodes_to_mask:
        layer = int(node.split(".")[0])
        head = int(node.split(".")[1])
        qkv = node.split(".")[2]

        def head_ablation_hook(value, hook):
            print(f"Shape of the value tensor: {value.shape}")
            value[:, :, head, :] = 0.0
            return value

        a_hook = (utils.get_act_name(qkv, int(layer)), head_ablation_hook)
        forward_hooks.append(a_hook)
    return forward_hooks


if __name__ == "__main__":
    from transformer_lens.HookedTransformer import (
        HookedTransformer,
        # MaskedHookedTransformer,
    )

    regularization_params = [
        # 1e4,
        # 1e3,
        1e2,
        # 1e1,
        # 1e0,
        # 0.1,
        # 0.01,
        # 0.001,
    ]
    is_masked = True
    logit_diff_list = []
    number_of_nodes_list = []

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
            sanity_check_with_transformer_lens(nodes_to_mask)

    wandb.init(project="pareto-subnetwork-probing", entity="acdcremix")
    import plotly.express as px
    import pandas as pd

    df = pd.DataFrame(
        {
            "x": number_of_nodes_list,
            "y": [i.cpu().detach().item() for i in logit_diff_list],
            "regularization_params": regularization_params,
        }
    )
    plt = px.scatter(df, x="x", y="y", hover_data=["regularization_params"])
    plt.update_layout(xaxis_title="Number of Nodes", yaxis_title="Logit Diff")
    wandb.log({"number_of_nodes": plt})
    wandb.finish()
