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
import plotly.graph_objects as go
import transformer_lens.utils as utils
import wandb
from interp.circuit.causal_scrubbing.dataset import Dataset
from tqdm import tqdm
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.hook_points import HookPoint, MaskedHookPoint
from transformer_lens.ioi_dataset import IOIDataset
from transformer_lens.utils import make_nd_dict
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


def log_plotly_bar_chart(x: List[str], y: List[float], name = "mask_scores") -> None:
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    wandb.log({name: fig})


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


if False: # I think the resampled points can and should be fixed ... ?
    def do_random_resample_caching(
        model: HookedTransformer, train_data: torch.Tensor
    ) -> HookedTransformer:
        for layer in model.blocks:
            layer.attn.hook_q.is_caching = True
            layer.attn.hook_k.is_caching = True
            layer.attn.hook_v.is_caching = True

        _ = model(train_data) # !!!

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

#%%

receivers_to_senders = make_nd_dict(list[tuple], n = 3)
all_senders = defaultdict(list[tuple])
all_senders["blocks.0.hook_resid_pre"] = [(None,)]

for layer_index in range(model.cfg.n_layers):
    for letter in "qkv":
        receiver_name = f"blocks.{layer_index}.hook_{letter}_input"
        for head_index in range(model.cfg.n_heads):
            receiver_slice_tuple = (None, None, head_index)
            for sender_name in all_senders:
                for sender_slice_tuple in all_senders[sender_name]:
                    # TODO make this sender tuple an object of its own for type reasons
                    receivers_to_senders[receiver_name][receiver_slice_tuple][sender_name].append(sender_slice_tuple)
    finally_the_sender = f"blocks.{layer_index}.attn.hook_result"
    for head_index in range(model.cfg.n_heads):
        all_senders[finally_the_sender].append((None, None, head_index))

receiver_name = f"blocks.{model.cfg.n_layers-1}.hook_resid_post"
receiver_slice_tuple = (None,)
for sender_name in all_senders:
    for sender_slice_tuple in all_senders[sender_name]:
        receivers_to_senders[receiver_name][receiver_slice_tuple][sender_name].append(sender_slice_tuple)

names_senders = list(all_senders.keys())

#%% 

def saver_hook(z, hook):
    print("SAVERHOOKING")
    hook.global_cache[hook.name] = z # crucial that this is THE TRUE THING so gradinets flooooow
    return z

all_senders

model.reset_hooks()
for name in names_senders:
    model.add_hook(
        name=name,
        hook=saver_hook,
    )

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

# do a mini test: if we use all the patched activations, what happens?
def editor_hook(z, hook):
    for receiver_tuple_slice in receivers_to_senders[hook.name]:
        receiver_slicer = create_slicer(receiver_tuple_slice)
        # assert False, "Change all these create slicer things into inline slices..."
        for sender_name in receivers_to_senders[hook.name][receiver_tuple_slice]:
            for sender_tuple_slice in receivers_to_senders[hook.name][receiver_tuple_slice][sender_name]:
                sender_slicer = create_slicer(sender_tuple_slice)
                hook.tens = receiver_slicer(z)
                hook.tens.retain_grad()
                print(hook.tens.norm(), "tens1", end=" ")
                hook.clean_part_factor = hook.global_cache.sampled_mask[hook.name][receiver_tuple_slice][sender_name][sender_tuple_slice]
                hook.clean_part_factor.retain_grad()
                hook.removed_clean_part = hook.clean_part_factor * sender_slicer(hook.global_cache[sender_name]) # don't do inplace things ughhhh but inplace things are punishing us! 
                hook.removed_clean_part.retain_grad()

                hook.tens = hook.tens - hook.removed_clean_part
                hook.corrupted_part_factor = 1 - hook.clean_part_factor
                hook.corrupted_part_factor.retain_grad()
                hook.tens = hook.tens + hook.corrupted_part_factor * sender_slicer(hook.global_cache.corrupt_cache[sender_name].to(hook.tens.device))
                hook.tens.retain_grad()
                print(hook.tens.norm(), "tens2", end=" ")

                if len(receiver_tuple_slice) == 1:
                    z = hook.tens
                else:
                    assert len(receiver_tuple_slice) == 3
                    assert None == receiver_tuple_slice[0] == receiver_tuple_slice[1]
                    z = torch.cat((z[:, :, :receiver_tuple_slice[2]], hook.tens.unsqueeze(2), z[:, :, receiver_tuple_slice[2]+1:]), dim=2) 
                    # shoutout to GPT-4 fo the above line

                # tens += sender_slicer(hook.global_cache.corrupt_cache[sender_name]).to(tens.device)
                warnings.warn("So far this must use the same tensor for what all heads send from a layer")

    print(hook.name, z.norm().item())
    return z

all_receivers_names = list(receivers_to_senders.keys())

hooks = []
for name in all_receivers_names:
    model.add_hook(
        name=name,
        hook=editor_hook,
    )

assert len(model.hook_dict[all_receivers_names[-1]].fwd_hooks) > 0, (all_receivers_names[-1], "should surely have had a hook added")

#%% [markdown]

# Begin a forward pass
# sample the parameter values

if False:
    out=model(train_data_tensor)

    assert list(out.shape) == list(bad_logits.shape), f"{out.shape} != {bad_logits.shape}"
    assert torch.allclose(out, bad_logits), (out.norm(), bad_logits.norm())

#%%

# commented out so we can access variables in notebook
# def train_induction(
#     model, mask_lr=0.01, epochs=30, verbose=True, lambda_reg=100,
# ):

if True: # in notebook not function
    mask_lr = 0.1
    warnings.warn("Turn mask_lr down once issue fixed...")
    epochs = 300
    lambda_reg = 100
    verbose = True

    from induction_utils import ct

    fcontents = ""
    with open(__file__, "r") as f:
        fcontent = f.read()

    run_name = f"run_{ct()}"
    wandb.init(
        name=run_name,
        project="subnetwork_probing_edges",
        entity="remix_school-of-rock", 
        config={"epochs": epochs, "mask_lr": mask_lr, "lambda_reg": lambda_reg},
        notes=fcontent,
    )
    (
        train_data_tensor,
        patch_data_tensor,
        dataset,
        _,
        _,
        mask_reshaped,
    ) = get_induction_dataset()

    # # one parameter per thing that is masked
    # mask_params = [
    #     p
    #     for n, p in model.named_parameters()
    #     if "mask_scores" in n and p.requires_grad
    # ]
    # # parameters for the probe (we don't use a probe)
    # model_params = [
    #     p
    #     for n, p in model.named_parameters()
    #     if "mask_scores" not in n and p.requires_grad
    # ]

    # with torch.set_grad_enabled(True):
    if True: # above didn't help
        mask_params = []
        for receiver_name in receivers_to_senders:
            for receiver_tuple_slice in receivers_to_senders[receiver_name]:
                for sender_name in receivers_to_senders[receiver_name][receiver_tuple_slice]:
                    for sender_slice_tuple in receivers_to_senders[receiver_name][receiver_tuple_slice][sender_name]:
                        parameter = torch.nn.Parameter(torch.zeros(1, requires_grad=True, device=model.global_cache.device))
                        parameter.retain_grad()
                        model.global_cache.parameters[receiver_name][receiver_tuple_slice][sender_name][sender_slice_tuple] = parameter
                        mask_params.append(parameter)
        print("Working with", len(mask_params), "parameters")

        # optimizer = torch.optim.Adam(parameters, lr=mask_lr)

        model.global_cache.init_weights()
        trainer = torch.optim.Adam(mask_params, lr=mask_lr)
        log = []
        model.train()
        torch.autograd.set_detect_anomaly(True)

        for epoch in tqdm(range(epochs)):  # tqdm.notebook.tqdm(range(epochs)):
            # model = do_random_resample_caching(model, patch_data_tensor)

            trainer.zero_grad()
            model.global_cache.sample_mask()

            # compute loss, also log other metrics
            # logit_diff_term = negative_log_probs(
            #     dataset, model(train_data_tensor), mask_reshaped
            # )
            
            warnings.warn("TODO: make sure to update to the KL divergence")

            if True:
                for n, p in model.named_parameters():
                    p.retain_grad()
                logits = model(train_data_tensor)
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
                masked_log_probs.retain_grad()
                result = (-1.0 * (masked_log_probs.sum())) / denom
                result.retain_grad()

                print("Result", result, denom)

                logit_diff_term = result
                # logit_diff_term.retain_grad()
                # return result

            else:
                logit_diff_term = negative_log_probs( # OK sure all loss named logit_diff
                    dataset, model(train_data_tensor), mask_reshaped
                )

            warnings.warn("TODO: add regularizer")
            regularizer_term = None
            # regularizer_term = regularizer(model)
            # loss = logit_diff_term + regularizer_term * lambda_reg
            loss = logit_diff_term
            loss.backward()
            wandb.log(
                {
                    "regularisation_loss": regularizer_term,
                    "KL_loss": logit_diff_term,
                    "total_loss": loss,
                }
            )
            trainer.step() # doesn't affect
            log.append({"loss_val": loss.item()})

            # for p in mask_params:
            #     pval = p.item()
            #     # show 2 DPs
            #     pval = round(pval, 2)
            #     print(pval, end=" ")

            # bar chart
            xs = []
            ms = []
            ps = []

            for recevier_name in model.global_cache.sampled_mask:
                for receiver_tuple_slice in model.global_cache.sampled_mask[recevier_name]:
                    for sender_name in model.global_cache.sampled_mask[recevier_name][receiver_tuple_slice]:
                        for sender_slice_tuple in model.global_cache.sampled_mask[recevier_name][receiver_tuple_slice][sender_name]:
                            # print(recevier_name, receiver_tuple_slice, sender_name, sender_slice_tuple)
                            m = model.global_cache.sampled_mask[recevier_name][receiver_tuple_slice][sender_name][sender_slice_tuple]
                            p = model.global_cache.parameters[recevier_name][receiver_tuple_slice][sender_name][sender_slice_tuple]
                            mval = m.item()
                            pval = p.item()

                            xs.append(recevier_name + " " + str(receiver_tuple_slice) + " " + sender_name + " " + str(sender_slice_tuple))
                            ms.append(mval)
                            ps.append(pval)

                            # show 2 DPs
                            mval = round(mval, 2)
                            print(mval, end=" ")
        
            log_plotly_bar_chart(xs, ms, name="mask_scores")
            log_plotly_bar_chart(xs, ps, name="parameter_values")

            warnings.warn("TODO: implement this ... ?")
            # if epoch % 10 == 0:
            #     number_of_nodes, nodes_to_mask = visualize_mask(model)


    wandb.finish()
    warnings.warn("Add back in return statements")
    # return log, model, number_of_nodes, logit_diff_term, nodes_to_mask

#%% [markdown]
# Rest of Augustine's file 

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

#%%

if __name__ == "__main__":
    model = get_induction_model()
    regularization_params = [ # TODO fix this ...
        700,
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