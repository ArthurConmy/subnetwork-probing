import wandb
import torch
from train import regularizer, logit_diff_from_ioi_dataset
from transformer_lens.ioi_dataset import IOIDataset
from transformer_lens.HookedTransformer import HookedTransformer
from typing import Dict, List, Tuple
from induction_utils import get_induction_model, get_induction_dataset, compute_no_edges_in_transformer_lens
from train_induction import kl_divergence, BASE_MODEL_PROBS

N = 100
NUMBER_OF_HEADS = 8
NUMBER_OF_LAYERS = 2


def get_gradients(
    induction_model,
    mask_lr=0.001,
    epochs=30,
    verbose=True,
    lambda_reg=100,
):
    (
        train_data_tensor,
        patch_data_tensor,
        dataset,
        _,
        _,
        mask_reshaped,
    ) = get_induction_dataset()

    # one parameter per thing that is masked
    mask_params = [
        p
        for n, p in induction_model.named_parameters()
        if "mask_scores" in n and p.requires_grad
    ]
    # parameters for the probe (we don't use a probe)
    model_params = [
        p
        for n, p in induction_model.named_parameters()
        if "mask_scores" not in n and p.requires_grad
    ]
    assert len(model_params) == 0, ("MODEL should be empty", model_params)
    trainer = torch.optim.Adam(mask_params, lr=mask_lr)
    induction_model.train()
    trainer.zero_grad()
    logit_diff_term = kl_divergence(
        dataset, induction_model(train_data_tensor), mask_reshaped
    )
    loss = logit_diff_term
    loss.backward()
    return induction_model


def log_plotly_bar_chart(x: List[str], y: List[float]) -> None:
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    wandb.log({"mask_scores": fig})


def compute_mask_scores(model: HookedTransformer) -> Dict[str, float]:
    node_name_list = []
    mask_scores_for_names = []
    total_nodes = 0
    mask_score_dict = {}
    for layer_index, layer in enumerate(model.blocks):
        for head_index in range(NUMBER_OF_HEADS):
            for q_k_v in ["q", "k", "v"]:
                total_nodes += 1
                if q_k_v == "q":
                    mask_score = (
                        layer.attn.hook_q.report_mask_importance()[head_index]
                        .cpu()
                        .item()
                    )
                if q_k_v == "k":
                    mask_score = (
                        layer.attn.hook_k.report_mask_importance()[head_index]
                        .cpu()
                        .item()
                    )
                if q_k_v == "v":
                    mask_score = (
                        layer.attn.hook_v.report_mask_importance()[head_index]
                        .cpu()
                        .item()
                    )
                node_name = f"layer_{layer_index}_head_{head_index}_{q_k_v}"
                node_name_list.append(node_name)
                mask_scores_for_names.append(
                    layer.attn.hook_v.mask_scores[head_index].cpu().item()
                )
                mask_score_dict[node_name] = mask_score

    mask_score_dict = normalize_mask_scores(mask_score_dict)
    assert len(mask_scores_for_names) == 3 * NUMBER_OF_HEADS * NUMBER_OF_LAYERS
    log_plotly_bar_chart(x=node_name_list, y=mask_scores_for_names)
    return mask_score_dict


def normalize_mask_scores(mask_score_dict):
    keys = list(mask_score_dict.keys())
    values = list(mask_score_dict.values())
    layer_lists = [[] for _ in range(NUMBER_OF_LAYERS)]

    for i, value in enumerate(values):
        layer = int(keys[i].split("_")[1])
        layer_lists[layer].append(value)

    for i, value in enumerate(values):
        layer = int(keys[i].split("_")[1])
        mask_score_dict[keys[i]] = (value - min(layer_lists[layer])) / (
            max(layer_lists[layer]) - min(layer_lists[layer])
        )
    return mask_score_dict


def mask_based_on_scores(model, scores_dict, number_of_heads_to_mask):
    heads_to_mask = scores_dict.keys()
    scores_list = scores_dict.values()
    heads_to_mask, scores_list = zip(
        *sorted(zip(heads_to_mask, scores_list), key=lambda x: x[1])
    )
    for i, score in enumerate(scores_list):
        if i < number_of_heads_to_mask:
            model = mask_head(model, heads_to_mask[i])
    nodes_to_mask = heads_to_mask[:number_of_heads_to_mask]
    return model, nodes_to_mask


def mask_head(model, head_name):
    q_k_v = head_name.split("_")[-1]
    layer = int(head_name.split("_")[1])
    head = int(head_name.split("_")[3])
    for layer_index, layer_object in enumerate(model.blocks):
        for head_index in range(NUMBER_OF_HEADS):
            if q_k_v == "q" and layer_index == layer and head_index == head:
                layer_object.attn.hook_q.mask_scores.requires_grad = False
                layer_object.attn.hook_q.mask_scores[head_index] = (
                    layer_object.attn.hook_q.mask_scores[head_index] * 0.0
                )
            if q_k_v == "k" and layer_index == layer and head_index == head:
                layer_object.attn.hook_k.mask_scores.requires_grad = False
                layer_object.attn.hook_k.mask_scores[head_index] = (
                    layer_object.attn.hook_k.mask_scores[head_index] * 0.0
                )
            if q_k_v == "v" and layer_index == layer and head_index == head:
                layer_object.attn.hook_v.mask_scores.requires_grad = False
                layer_object.attn.hook_v.mask_scores[head_index] = (
                    layer_object.attn.hook_v.mask_scores[head_index] * 0.0
                )

    return model


def main():
    wandb.init(
        project="better_than_one",
        entity="remix_school-of-rock",
    )
    kl_list = []
    heads_masked = []
    number_of_edges = []
    for i in range(3 * NUMBER_OF_HEADS * NUMBER_OF_LAYERS):
        induction_model = get_induction_model()
        induction_model.freeze_weights()

        (
            train_data_tensor,
            rand_data_tensor,
            dataset,
            _,
            _,
            mask_reshaped,
        ) = get_induction_dataset()

        assert induction_model.blocks[0].attn.hook_q.second_cache is None
        induction_model(rand_data_tensor) # save to the second cache random activations
        assert induction_model.blocks[0].attn.hook_q.second_cache is not None

        induction_model = get_gradients(induction_model, lambda_reg=100)
        mask_scores_dict = compute_mask_scores(model=induction_model)
        kl = kl_divergence(dataset, induction_model(train_data_tensor), mask_reshaped)
        print("kl before", kl)
        induction_model, nodes_to_mask = mask_based_on_scores(
            model=induction_model,
            scores_dict=mask_scores_dict,
            number_of_heads_to_mask=i,
        )
        curr_number_of_edges = compute_no_edges_in_transformer_lens(nodes_to_mask)
        number_of_edges.append(curr_number_of_edges)
        kl = kl_divergence(dataset, induction_model(train_data_tensor), mask_reshaped)
        print("kl after", kl)
        kl_list.append(kl.cpu().item())
        heads_masked.append(i)
        # log a plotly scatter plot
        import plotly.express as px
        import pandas as pd

        # create a sample dataframe
        data = pd.DataFrame(
            {
                "x": number_of_edges,
                "y": kl_list,
            }
        )

        # create the scatter plot
        fig = px.scatter(data, x="x", y="y")

        wandb.log({"scatter": fig})


if __name__ == "__main__":
    main()
