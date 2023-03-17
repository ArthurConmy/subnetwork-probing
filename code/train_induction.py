#%%
from copy import deepcopy
from functools import partial
from typing import Dict, List
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd
import torch
import warnings
import torch.nn.functional as F
import transformer_lens.utils as utils
import wandb
from interp.circuit.causal_scrubbing.dataset import Dataset
from tqdm import tqdm
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.hook_points import HookPoint, MaskedHookPoint
from transformer_lens.ioi_dataset import IOIDataset
from induction_utils import (
    get_induction_dataset,
    get_induction_model,
    compute_no_edges_in_transformer_lens,
)
from base_probs import compute_base_model_probs

SEQ_LEN = 300
NUM_EXAMPLES = 40
NUMBER_OF_HEADS = 8
NUMBER_OF_LAYERS = 2
BASE_MODEL_PROBS = compute_base_model_probs()
# don't want to backprop through this
BASE_MODEL_PROBS = BASE_MODEL_PROBS.detach()


def log_plotly_bar_chart(x: List[str], y: List[float]) -> None:
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    wandb.log({"mask_scores": fig})


def visualize_mask(model: HookedTransformer) -> None:
    node_name_list = []
    mask_scores_for_names = []
    total_nodes = 0
    nodes_to_mask = []
    for layer_index, layer in enumerate(model.blocks):
        for head_index in range(NUMBER_OF_HEADS):
            for q_k_v in ["q", "k", "v"]:
                total_nodes += 1
                if q_k_v == "q":
                    mask_sample = (
                        layer.attn.hook_q.sample_mask()[head_index].cpu().item()
                    )
                if q_k_v == "k":
                    mask_sample = (
                        layer.attn.hook_k.sample_mask()[head_index].cpu().item()
                    )
                if q_k_v == "v":
                    mask_sample = (
                        layer.attn.hook_v.sample_mask()[head_index].cpu().item()
                    )
                node_name = f"layer_{layer_index}_head_{head_index}_{q_k_v}"
                node_name_list.append(node_name)
                mask_scores_for_names.append(
                    layer.attn.hook_v.mask_scores[head_index].cpu().item()
                )
                if mask_sample < 0.5:
                    nodes_to_mask.append(node_name)

    assert len(mask_scores_for_names) == 3 * NUMBER_OF_HEADS * NUMBER_OF_LAYERS
    log_plotly_bar_chart(x=node_name_list, y=mask_scores_for_names)
    node_count = total_nodes - len(nodes_to_mask)
    return node_count, nodes_to_mask


def regularizer(
    model: HookedTransformer,
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
        for (n, p) in model.named_parameters()
        if "mask_scores" in n
    ]
    return torch.mean(torch.stack(mask_scores))


def negative_log_probs(
    dataset: Dataset, logits: torch.Tensor, mask_reshaped: torch.Tensor
) -> float:
    """NOTE: this average over all sequence positions, I'm unsure why..."""
    labels = dataset.arrs["labels"].evaluate()
    probs = F.softmax(logits, dim=-1)

    assert probs.min() >= 0.0
    assert probs.max() <= 1.0

    log_probs = probs[
        torch.arange(NUM_EXAMPLES).unsqueeze(-1),
        torch.arange(SEQ_LEN).unsqueeze(0),
        labels,
    ].log()

    assert mask_reshaped.shape == log_probs.shape, (
        mask_reshaped.shape,
        log_probs.shape,
    )
    denom = mask_reshaped.int().sum().item()

    masked_log_probs = log_probs * mask_reshaped.int()
    result = (-1.0 * (masked_log_probs.sum())) / denom

    print("Result", result, denom)
    return result


def kl_divergence(dataset: Dataset, logits: torch.Tensor, mask_reshaped: torch.Tensor):
    """Compute KL divergence between base_model_probs and probs, taken from Arthur's ACDC code"""
    # labels = dataset.arrs["labels"].evaluate()
    probs = F.softmax(logits, dim=-1)

    assert probs.min() >= 0.0
    assert probs.max() <= 1.0

    denom = mask_reshaped.int().sum().item()
    kl_div = (BASE_MODEL_PROBS * (BASE_MODEL_PROBS.log() - probs.log())).sum(dim=-1)
    kl_div = kl_div * mask_reshaped.int()

    return kl_div.sum() / denom


def do_random_resample_caching(
    model: HookedTransformer, train_data: torch.Tensor
) -> HookedTransformer:
    for layer in model.blocks:
        layer.attn.hook_q.is_caching = True
        layer.attn.hook_k.is_caching = True
        layer.attn.hook_v.is_caching = True

    _ = model(train_data)

    for layer in model.blocks:
        layer.attn.hook_q.is_caching = False
        layer.attn.hook_k.is_caching = False
        layer.attn.hook_v.is_caching = False

    return model

#%% [markdown]

# will become train edge induction, but for now prototype in notebook
model = get_induction_model()
model.set_use_split_qkv_input(True)
model.set_use_attn_result(True)
model.cfg

#%%

# turn off Augustine's node subnetwork probing
for hp in model.hook_dict.values():
    if isinstance(hp, MaskedHookPoint):
        hp.is_disabled = True

#%%

# verify that:
for hp in model.hook_dict.values():
    if isinstance(hp, MaskedHookPoint):
        assert hp.is_disabled

#%%

(
    train_data_tensor,
    patch_data_tensor,
    dataset,
    _,
    _,
    mask_reshaped,
) = get_induction_dataset()
#%%

corrupt_cache = {}
model.cache_all(corrupt_cache)

def norm_peeker(z, hook):
    print(hook.name, z.norm().item())
    return z
for name in model.hook_dict:
    model.add_hook(name, norm_peeker)
bad_logits = model(patch_data_tensor)
model.reset_hooks()

model.global_cache.corrupt_cache = {k: v.cpu() for k, v in corrupt_cache.items()}
# del corrupt_cache
# torch.cuda.empty_cache()

#%%

receivers_to_senders = OrderedDict() # TODO typehint
all_senders = [
    ("blocks.0.hook_resid_pre", (None,)),
]

def create_slicer(tup):
    def slicer(z):
        if len(tup) == 1:
            if tup[0] is None:
                return z
            else:
                return z[tup[0]]
        
        elif len(tup) == 2:
            assert tup[0] is None
            return z[:, tup[1]]

        elif len(tup) == 3:
            assert tup[0] is None
            assert tup[1] is None
            return z[:, :, tup[2]]

        elif len(tup) == 4:
            assert tup[0] is None
            assert tup[1] is None
            assert tup[2] is None
            return z[:, :, :, tup[3]]
        
        else:
            raise ValueError("tup too long", f"{len(tup)=}", f"{tup=}")

    return slicer

for layer_index in range(model.cfg.n_layers):
    for head_index in range(model.cfg.n_heads):
        for letter in "qkv":
            receiver_name = f"blocks.{layer_index}.hook_{letter}_input"
            receiver_slice_tuple = (None, None, head_index)
            receivers_to_senders[receiver_name] = {}
            receivers_to_senders[receiver_name][receiver_slice_tuple] = []
            for sender in all_senders:
                receivers_to_senders[receiver_name][receiver_slice_tuple].append(sender)
    finally_the_sender = f"blocks.{layer_index}.attn.hook_result"
    for head_index in range(model.cfg.n_heads):
        all_senders.append((finally_the_sender, (None, None, head_index)))

receiver_name = f"blocks.{model.cfg.n_layers-1}.hook_resid_post"
receiver_slice_tuple = (None,)
receivers_to_senders[receiver_name] = {}
for sender in all_senders:
    receivers_to_senders[receiver_name][receiver_slice_tuple] = [sender] # .append(sender)

#%%

names_senders = list(set([name for name, _ in all_senders]))

# def saver_hook(z, hook):
#     print("saverhooking")
#     hook.global_cache[hook.name] = z
#     return z

# for name in names_senders:
#     model.add_hook(
#         name=name,
#         hook=saver_hook,
#     )

#%%

# test run to see what happens, do things actually save
td = model(train_data_tensor)

#%% 

# do a mini test: if we use all the patched activations, what happens?
def editor_hook(z, hook):
    curz = z.clone() 

    for receiver_tuple_slice in receivers_to_senders[hook.name]:
        receiver_slicer = create_slicer(receiver_tuple_slice)
        for sender_name, sender_tuple_slice in receivers_to_senders[hook.name][receiver_tuple_slice]:
            sender_slicer = create_slicer(sender_tuple_slice)
            warnings.warn("Add this back when you have saving_hook working")
            # curz[slice_slice] -= hook.global_cache[sender_name][slice_slice]

            try:
                tens = receiver_slicer(curz)
                tens += sender_slicer(hook.global_cache.corrupt_cache[sender_name]).to(tens.device)
                warnings.warn("So far this must use the same tensor for what all heads send from a layer")
            except:
                import IPython; IPython.embed()

    print(hook.name, curz.norm().item())
    return curz

all_receivers_names = list(receivers_to_senders.keys())
names_senders = ['blocks.0.attn.hook_result',
'blocks.1.attn.hook_result',
'blocks.0.hook_resid_pre']
hooks=[]

model.reset_hooks()
for name in all_receivers_names:
    # hooks.append((name, editor_hook))
    model.add_hook(
        name=name,
        hook=editor_hook,
    )

assert len(model.hook_dict[all_receivers_names[-1]].fwd_hooks) > 0, (all_receivers_names[-1], "should surely have had a hook added")

#%%

out=model(train_data_tensor)

# out = model.run_with_hooks(
#     train_data_tensor,
#     fwd_hooks=hooks,
# )

assert False, "WHY IS THERE NO PRINTING OUT OF NORMS HERE???"

#%%

assert list(out.shape) == list(bad_logits.shape), f"{out.shape} != {bad_logits.shape}"
assert torch.allclose(out, bad_logits), (out.norm(), bad_logits.norm())

#%%

def train_induction(
    model, mask_lr=0.01, epochs=30, verbose=True, lambda_reg=100,
):

    wandb.init(
        project="subnetwork-probing",
        entity="remix_school-of-rock", 
        config={"epochs": epochs, "mask_lr": mask_lr, "lambda_reg": lambda_reg},
    )
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
        for n, p in model.named_parameters()
        if "mask_scores" in n and p.requires_grad
    ]
    # parameters for the probe (we don't use a probe)
    model_params = [
        p
        for n, p in model.named_parameters()
        if "mask_scores" not in n and p.requires_grad
    ]
    assert len(model_params) == 0, ("MODEL should be empty", model_params)
    trainer = torch.optim.Adam(mask_params, lr=mask_lr)
    log = []

    for epoch in tqdm(range(epochs)):  # tqdm.notebook.tqdm(range(epochs)):
        model = do_random_resample_caching(model, patch_data_tensor)
        model.train()
        trainer.zero_grad()
        # compute loss, also log other metrics
        # logit_diff_term = negative_log_probs(
        #     dataset, model(train_data_tensor), mask_reshaped
        # )
        logit_diff_term = kl_divergence(
            dataset, model(train_data_tensor), mask_reshaped
        )
        regularizer_term = regularizer(model)
        loss = logit_diff_term + regularizer_term * lambda_reg
        loss.backward()

        wandb.log(
            {
                "regularisation_loss": regularizer_term,
                "KL_loss": logit_diff_term,
                "total_loss": loss,
            }
        )
        trainer.step()

        log.append({"loss_val": loss.item()})

        if epoch % 10 == 0:
            number_of_nodes, nodes_to_mask = visualize_mask(model)
    # wandb.finish()
    # torch.save(model.state_dict(), "masked_model.pt")
    return log, model, number_of_nodes, logit_diff_term, nodes_to_mask


# check regularizer can set all the
def sanity_check_with_transformer_lens(mask_dict):
    ioi_dataset = IOIDataset(prompt_type="ABBA", N=N, nb_templates=1)
    train_data = ioi_dataset.toks.long()
    model = HookedTransformer.from_pretrained(is_masked=False, model_name="model")
    model.freeze_weights()
    logits = model(train_data)
    logit_diff = logit_diff_from_ioi_dataset(logits, train_data, mean=True)

    fwd_hooks = make_forward_hooks(mask_dict)
    logits = model.run_with_hooks(train_data, return_type="logits", fwd_hooks=fwd_hooks)
    logit_diff_masked = logit_diff_from_ioi_dataset(logits, train_data, mean=True)
    print("original logit diff", logit_diff)
    print("masked logit diff", logit_diff_masked)


def make_forward_hooks(mask_dict):
    forward_hooks = []
    for layer in range(NUMBER_OF_LAYERS):
        for head in range(NUMBER_OF_HEADS):
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


def get_nodes_mask_dict(model: HookedTransformer):
    mask_value_dict = {}
    for layer_index, layer in enumerate(model.blocks):
        for head_index in range(NUMBER_OF_HEADS):
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
    model = get_induction_model()
    regularization_params = [
        # 1e-2,
        # 1e-1,
        # 1e1,
        # 20,
        # 40,
        # 50,
        # 55,
        # 60,
        # 65,
        # 70,
        # 80,
        # 100,
        # 120,
        # 140,
        # 160,
        # 180,
        # 200,
        # 250,
        # 300,
        # 310,
        # 320,
        # 330,
        # 350,
        # 360,
        # 370,
        # 380,
        # 400,
        # 500,
        # 600,
        700,
        # 800,
        # 900,
        # 1e3,
    ]

    is_masked = True
    logit_diff_list = []
    number_of_nodes_list = []
    percentage_binary_list = []
    number_of_edges = []

    for a_regulation_param in regularization_params:
        for task in ["IOI"]:
            model.freeze_weights()
            print("Finding subnetwork...")
            assert task == "IOI"
            log, model, number_of_nodes, logit_diff, nodes_to_mask = train_induction(
                deepcopy(model), lambda_reg=a_regulation_param
            )
            print("nodes to mask", nodes_to_mask)
            logit_diff_list.append(logit_diff)
            number_of_nodes_list.append(number_of_nodes)
            mask_val_dict = get_nodes_mask_dict(model)
            percentage_binary = log_percentage_binary(mask_val_dict)
            number_of_edges.append(compute_no_edges_in_transformer_lens(nodes_to_mask))
            wandb.log({"percentage_binary": percentage_binary})
            percentage_binary_list.append(percentage_binary)
            # sanity_check_with_transformer_lens(mask_val_dict)
            wandb.finish()

    # make sure that regularizer can be optimized DONE
    # make sure logit diff can be optimized DONE
    # make sure that the mask is the right shape HOLD
    # reimplement all the diagnostics that are commented out TODO
    # reimplement sanity  check with transformer lens TODO
    # make sure that the input data makes sense
    # make sure that the model makes correct predictions
    # brainstorm more
    #
    wandb.init(project="pareto-subnetwork-probing", entity="remix_school-of-rock")
    import plotly.express as px

    df = pd.DataFrame(
        {
            "x": np.log(number_of_edges),
            "y": [i.detach().cpu().item() for i in logit_diff_list],
            "regularization_params": regularization_params,
            "percentage_binary": percentage_binary_list,
        }
    )
    plt = px.scatter(
        df, x="x", y="y", hover_data=["regularization_params", "percentage_binary"]
    )
    plt.update_layout(xaxis_title="Number of Nodes", yaxis_title="KL")
    wandb.log({"number_of_nodes": plt})
    wandb.finish()
#%%