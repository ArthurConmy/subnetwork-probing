# %%
import pickle
from typing import Tuple

import os
import einops
import rust_circuit as rc
import torch
import datetime
from interp.circuit.causal_scrubbing.dataset import Dataset
from interp.circuit.interop_rust.model_rewrites import To, configure_transformer
from interp.circuit.interop_rust.module_library import load_model_id
from interp.circuit.projects.gpt2_gen_induction.rust_path_patching import make_arr
from interp.tools.data_loading import get_val_seqs
from interp.tools.indexer import TORCH_INDEXER as I
from interp.tools.rrfs import RRFS_DIR
from transformer_lens import HookedTransformer
from typing import Dict
from interp.circuit.causal_scrubbing.dataset import Dataset


def get_induction_dataset():
    try:
        with open(
            os.path.expanduser(
                "~/induction/data/masks/mask_repeat_candidates.pkl"
            ),
            "rb",
        ) as f:
            mask_repeat_candidates = pickle.load(f)
            # t[1] has 132, 136, 176 available...
            # if induction is AB...AB, B tokens are on 133(OK...), 137, 177
            # OK so this is where we punish losses
    except:
        raise Exception(
            "Have you cloned https://github.com/aryamanarora/induction ??? It is where all the masks are kept !!!"
        )

    DEVICE = "cuda:0"
    SEQ_LEN = 300
    NUM_EXAMPLES = 40
    MODEL_ID = "attention_only_2"
    n_files = 1
    reload_dataset = False
    toks_int_values: rc.Array

    (circ_dict, tokenizer, model_info) = load_model_id(MODEL_ID)
    if reload_dataset:
        dataset_toks = torch.tensor(
            get_val_seqs(n_files=n_files, files_start=0, max_size=SEQ_LEN + 1)
        ).cuda()
        NUM_EXAMPLES, _ = dataset_toks.shape
        toks_int_values = rc.Array(dataset_toks.float(), name="toks_int_vals")
        print(f'new dataset "{toks_int_values.repr()}"')
    else:
        P = rc.Parser()
        toks_int_values_raw = P(
            f"'toks_int_vals' [104091,301] Array 3f36c4ca661798003df14994"
        ).cast_array()

    CACHE_DIR = f"{RRFS_DIR}/ryan/induction_scrub/cached_vals"
    good_induction_candidate = torch.load(
        f"{CACHE_DIR}/induction_candidates_2022-10-15 04:48:29.970735.pt"
    ).to(device=DEVICE, dtype=torch.float32)
    assert (
        toks_int_values_raw.shape[0] >= SEQ_LEN
    ), f"toks_int_values_raw.shape[0] = {toks_int_values_raw.shape[0]} < {SEQ_LEN} - you could try increasing `n_files`"
    assert (
        toks_int_values_raw.shape[1] >= SEQ_LEN + 1
    ), f"toks_int_values_raw.shape[1] = {toks_int_values_raw.shape[1]} < {SEQ_LEN + 1}"

    tokens_device_dtype = rc.TorchDeviceDtype(device="cuda", dtype="int64")

    toks_int_values = make_arr(
        toks_int_values_raw.value[:NUM_EXAMPLES, :SEQ_LEN],
        name="toks_int_vals",
        device_dtype=tokens_device_dtype,
    )
    mask_reshaped = mask_repeat_candidates[
        :NUM_EXAMPLES, :SEQ_LEN
    ]  # only used for loss
    denom = mask_reshaped.int().sum().item()
    print("We're going to study", denom, "examples...")
    assert denom == 172, (denom, "was not expected")

    toks_int_labels = make_arr(
        toks_int_values_raw.value[:NUM_EXAMPLES, 1 : SEQ_LEN + 1],
        name="toks_int_labels",
        device_dtype=tokens_device_dtype,
    )

    def shuffle_tensor(tens):
        """Shuffle tensor along first dimension"""
        return tens[torch.randperm(tens.shape[0])]

    toks_int_values_other = make_arr(
        shuffle_tensor(toks_int_values.value[:NUM_EXAMPLES, : SEQ_LEN + 1]),
        name="toks_int_vals_other",
        device_dtype=tokens_device_dtype,
    )

    toks = tokenizer.batch_decode(
        good_induction_candidate.nonzero().flatten().view(-1, 1)
    )
    maxlen_tok = max((len(tok), tok) for tok in toks)

    circ_dict = {
        s: rc.cast_circuit(c, rc.TorchDeviceDtypeOp(device="cuda"))
        for s, c in circ_dict.items()
    }

    default_input = toks_int_values.rename("tokens")
    default_output = toks_int_labels.rename("labels")

    print("\ntokens of input and output")
    print(tokenizer.batch_decode(default_input.evaluate()[0, :10]))
    print(tokenizer.batch_decode(default_output.evaluate()[0, :10]))

    patch_input = toks_int_values_other.rename("tokens")
    patch_output = default_output  # oo cares..

    default_ds = Dataset({"tokens": default_input, "labels": default_output})
    patched_ds = Dataset({"tokens": patch_input, "labels": patch_output})
    return (
        default_ds.tokens.evaluate(),
        patched_ds.tokens.evaluate(),
        default_ds,
        circ_dict,
        model_info,
        mask_reshaped,
    )


def get_induction_model() -> HookedTransformer:
    SEQ_LEN = 300
    tokens_device_dtype = rc.TorchDeviceDtype(device="cuda", dtype="int64")

    _, _, default_ds, circ_dict, model_info, _ = get_induction_dataset()
    orig_circuit = circ_dict["t.bind_w"]
    tok_embeds = circ_dict["t.w.tok_embeds"]
    pos_embeds = circ_dict["t.w.pos_embeds"]

    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer("Hello world")["input_ids"]

    assert (
        model_info.causal_mask
    ), "Should not apply causal mask if the transformer doesn't expect it!"

    # TODO: may be an issue later!
    causal_mask = rc.Array(
        (torch.arange(SEQ_LEN)[:, None] >= torch.arange(SEQ_LEN)[None, :]).to(
            tok_embeds.cast_array().value
        ),
        f"t.a.c.causal_mask",
    )
    assert model_info.pos_enc_type == "shortformer"
    pos_embeds = pos_embeds.index(I[:SEQ_LEN], name="t.w.pos_embeds_idxed")

    tokens_arr = rc.cast_circuit(
        rc.Array(torch.zeros(SEQ_LEN).to(torch.long), name="tokens"),
        device_dtype=tokens_device_dtype.op(),
    )
    idxed_embeds = rc.GeneralFunction.gen_index(
        tok_embeds, tokens_arr, index_dim=0, name="idxed_embeds"
    )

    # CHECK
    model = rc.module_new_bind(
        orig_circuit,
        ("t.input", idxed_embeds),
        ("a.mask", causal_mask),
        ("a.pos_input", pos_embeds),
        name="t.call",
    )

    # CHECK
    model = model.update(
        "t.bind_w",
        lambda c: configure_transformer(
            c,
            To.ATTN_HEAD_MLP_NORM,
            split_by_head_config="full",
            use_pull_up_head_split=True,
            use_flatten_res=True,
            flatten_components=True,
        ),
    )
    model = model.cast_module().substitute()
    model.print_html()
    model = rc.Index(model, I[0]).rename("model")
    model = rc.conform_all_modules(model)
    model = model.update("t.call", lambda c: c.rename("logits"))
    model = model.update("t.call", lambda c: c.rename("logits_with_bias"))
    model = model.update(
        rc.Regex("[am]\\d(.h\\d)?$"), lambda c: c.rename(c.name + ".inner")
    )
    model = model.update("t.inp_tok_pos", lambda c: c.rename("embeds"))
    model = model.update("t.a.mask", lambda c: c.rename("padding_mask"))
    for l in range(model_info.params.num_layers):
        for h in range(8):
            model = model.update(f"b{l}.a.h{h}", lambda c: c.rename(f"a{l}.h{h}"))
        next = "final" if l == model_info.params.num_layers - 1 else f"a{l + 1}"
        model = model.update(f"b{l}", lambda c: c.rename(f"{next}.input"))

    def create_path_matcher(
        start_node: rc.MatcherIn, path: list[str], max_distance=10
    ) -> rc.IterativeMatcher:
        """
        Creates a matcher that matches a path of nodes, given in a list of names, where the maximum distance between each node on the path is max_distance
        """

        initial_matcher = rc.IterativeMatcher(start_node)
        max_dis_path_matcher = lambda name: rc.restrict(
            rc.Matcher(name), end_depth=max_distance
        )
        chain_matcher = initial_matcher.chain(max_dis_path_matcher(path[0]))
        for i in range(1, len(path)):
            chain_matcher = chain_matcher.chain(max_dis_path_matcher(path[i]))
        return chain_matcher

    q_path = [
        "a.comb_v",
        "a.attn_probs",
        "a.attn_scores",
        "a.attn_scores_raw",
        "a.q",
    ]
    k_path = [
        "a.comb_v",
        "a.attn_probs",
        "a.attn_scores",
        "a.attn_scores_raw",
        "a.k",
    ]
    v_path = ["a.comb_v", "a.v"]
    qkv_paths = {"q": q_path, "k": k_path, "v": v_path}
    attention_head_name = "a{layer}.h{head}"
    qkv_node_name = "a{layer}.h{head}.{qkv}"
    embed_name = "idxed_embeds"
    root_name = "final.input"
    no_layers = 2
    no_heads = 8
    new_circuit = model
    for l in range(no_layers):
        for h in range(no_heads):
            for qkv in ["q", "k", "v"]:
                qkv_matcher = create_path_matcher(f"a{l}.h{h}", qkv_paths[qkv])
                new_circuit = new_circuit.update(
                    qkv_matcher, lambda c: c.rename(f"a{l}.h{h}.{qkv}")
                )

    printer = rc.PrintHtmlOptions(
        shape_only_when_necessary=False,
        traversal=rc.restrict(
            rc.IterativeMatcher(
                "embeds", "padding_mask", "final.norm", rc.Regex("^[am]\\d(.h\\d)?$")
            ),
            term_if_matches=True,
        ),
    )
    new_circuit = rc.substitute_all_modules(new_circuit)
    new_circuit.get_unique("final.input").print_html()
    new_circuit = new_circuit.get_unique("logits")

    tokens_arr = make_arr(
        torch.arange(300), name="tokens", device_dtype=tokens_device_dtype,
    )

    new_circuit = new_circuit.update(rc.Matcher("tokens"), lambda _: tokens_arr,)

    new_logits = new_circuit.evaluate()
    print(torch.norm(new_logits))

    """
    Now get on to porting this to TransformerLens
    """

    import transformer_lens
    from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=256,
        n_ctx=2048,  # chekc pos embed size
        n_heads=8,
        d_head=32,
        # model_name : str = "custom"
        # d_mlp: Optional[int] = None
        # act_fn: Optional[str] = None
        d_vocab=50259,
        # eps: float = 1e-5
        use_attn_result=True,
        use_attn_scale=True,  # divide by sqrt(d_head)
        # use_local_attn: bool = False
        # original_architecture: Optional[str] = None
        # from_checkpoint: bool = False
        # checkpoint_index: Optional[int] = None
        # checkpoint_label_type: Optional[str] = None
        # checkpoint_value: Optional[int] = None
        # tokenizer_name: Optional[str] = None
        # window_size: Optional[int] = None
        # attn_types: Optional[List] = None
        # init_mode: str = "gpt2"
        # normalization_type: Optional[str] = "LN"
        # device: Optional[str] = None
        # attention_dir: str = "causal"
        attn_only=True,
        # seed: Optional[int] = None
        # initializer_range: float = -1.0
        # init_weights: bool = True
        # scale_attn_by_inverse_layer_idx: bool = False
        positional_embedding_type="shortformer",
        # final_rms: bool = False
        # d_vocab_out: int = -1
        # parallel_attn_mlp: bool = False
        # rotary_dim: Optional[int] = None
        # n_params: Optional[int] = None
        # use_hook_tokens: bool = False
    )

    et_model = transformer_lens.HookedTransformer(cfg, is_masked=True)

    # embed.W_E torch.Size([50259, 256]) True
    et_model.embed.W_E.data = new_circuit.get_unique("t.w.tok_embeds").evaluate()

    # pos_embed.W_pos torch.Size([2048, 256]) True
    et_model.pos_embed.W_pos.data = new_circuit.get_unique("t.w.pos_embeds").evaluate()

    # blocks.0.ln1.w torch.Size([256]) True
    et_model.blocks[0].ln1.w.data = new_circuit.get_unique("a0.ln.w.scale").evaluate()

    # blocks.0.ln1.b torch.Size([256]) True
    et_model.blocks[0].ln1.b.data = new_circuit.get_unique("a0.ln.w.bias").evaluate()

    # blocks.0.attn.W_Q torch.Size([8, 256, 32]) True
    et_model.blocks[0].attn.W_Q.data = einops.rearrange(
        new_circuit.get_unique("a0.w.q").evaluate(), "a b c -> a c b"
    )

    # blocks.0.attn.W_K torch.Size([8, 256, 32]) True
    et_model.blocks[0].attn.W_K.data = einops.rearrange(
        new_circuit.get_unique("a0.w.k").evaluate(), "a b c -> a c b"
    )

    # blocks.0.attn.W_V torch.Size([8, 256, 32]) True
    et_model.blocks[0].attn.W_V.data = einops.rearrange(
        new_circuit.get_unique("a0.w.v").evaluate(), "a b c -> a c b"
    )  # .reshape(8, 256, 32)

    # blocks.0.attn.W_O torch.Size([8, 32, 256]) True
    et_model.blocks[0].attn.W_O.data = einops.rearrange(
        new_circuit.get_unique("a0.w.o").evaluate(), "a b c -> a c b"
    )

    # blocks.0.attn.b_Q torch.Size([8, 32]) True
    et_model.blocks[0].attn.b_Q.data *= 0.0

    # blocks.0.attn.b_K torch.Size([8, 32]) True
    et_model.blocks[0].attn.b_K.data *= 0.0

    # blocks.0.attn.b_V torch.Size([8, 32]) True
    et_model.blocks[0].attn.b_V.data *= 0.0

    # blocks.0.attn.b_O torch.Size([256]) True
    et_model.blocks[0].attn.b_O.data *= 0.0

    # blocks.1.ln1.w torch.Size([256]) True
    et_model.blocks[1].ln1.w.data = new_circuit.get_unique("a1.ln.w.scale").evaluate()

    # blocks.1.ln1.b torch.Size([256]) True
    et_model.blocks[1].ln1.b.data = new_circuit.get_unique("a1.ln.w.bias").evaluate()

    # blocks.1.attn.W_Q torch.Size([8, 256, 32]) True
    et_model.blocks[1].attn.W_Q.data = einops.rearrange(
        new_circuit.get_unique("a1.w.q").evaluate(), "a b c -> a c b"
    )

    # blocks.1.attn.W_K torch.Size([8, 256, 32]) True
    et_model.blocks[1].attn.W_K.data = einops.rearrange(
        new_circuit.get_unique("a1.w.k").evaluate(), "a b c -> a c b"
    )

    # blocks.1.attn.W_V torch.Size([8, 256, 32]) True
    et_model.blocks[1].attn.W_V.data = einops.rearrange(
        new_circuit.get_unique("a1.w.v").evaluate(), "a b c -> a c b"
    )

    # blocks.1.attn.W_O torch.Size([8, 32, 256]) True
    et_model.blocks[1].attn.W_O.data = einops.rearrange(
        new_circuit.get_unique("a1.w.o").evaluate(), "a b c -> a c b"
    )

    # blocks.1.attn.b_Q torch.Size([8, 32]) True
    et_model.blocks[1].attn.b_Q.data *= 0.0

    # blocks.1.attn.b_K torch.Size([8, 32]) True
    et_model.blocks[1].attn.b_K.data *= 0.0

    # blocks.1.attn.b_V torch.Size([8, 32]) True
    et_model.blocks[1].attn.b_V.data *= 0.0

    # blocks.1.attn.b_O torch.Size([256]) True
    et_model.blocks[1].attn.b_O.data *= 0.0

    # ln_final.w torch.Size([256]) True
    et_model.ln_final.w.data = new_circuit.get_unique("final.ln.w.scale").evaluate()

    # ln_final.b torch.Size([256]) True
    et_model.ln_final.b.data = new_circuit.get_unique("final.ln.w.bias").evaluate()

    # unembed.W_U torch.Size([256, 50259]) True
    et_model.unembed.W_U.data = new_circuit.get_unique("t.w.unembed").evaluate().T

    # unembed.b_U torch.Size([50259]) True
    et_model.unembed.b_U.data *= 0.0
    return et_model


def get_ACDC_correspondece():

    """
    Edited version of the hierarchy tutorial that runs a number of threshold runs, and then uses the results to plot a pareto frontier.
    """

    import os
    import time
    import sys
    import pandas as pd
    import uuid
    from typing import Optional, Tuple
    import interp.tools.optional as op
    import numpy as np
    import rust_circuit as rc
    import torch
    from interp.circuit.causal_scrubbing.experiment import (
        Experiment,
        ExperimentCheck,
        ExperimentEvalSettings,
        ScrubbedExperiment,
    )
    import warnings
    from interp.circuit.causal_scrubbing.hypothesis import (
        Correspondence,
        CondSampler,
        ExactSampler,
        FuncSampler,
        InterpNode,
        UncondSampler,
        chain_excluding,
        corr_root_matcher,
    )
    import pandas as pd
    from interp.circuit.interop_rust.algebric_rewrite import (
        residual_rewrite,
        split_to_concat,
    )
    from interp.circuit.interop_rust.model_rewrites import To, configure_transformer
    from interp.circuit.interop_rust.module_library import load_model_id
    from interp.tools.indexer import TORCH_INDEXER as I
    from torch.nn.functional import binary_cross_entropy_with_logits
    import torch.nn.functional as F
    import wandb
    import datetime

    import IPython

    if IPython.get_ipython() is not None:
        IPython.get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
        IPython.get_ipython().run_line_magic("autoreload", "2")  # type: ignore
    from copy import deepcopy
    from typing import List
    from tqdm import tqdm
    from interp.tools.data_loading import get_val_seqs
    from interp.circuit.causal_scrubbing.dataset import Dataset
    from interp.circuit.causal_scrubbing.hypothesis import corr_root_matcher
    from interp.circuit.interop_rust.model_rewrites import To, configure_transformer
    from interp.circuit.interop_rust.module_library import load_model_id
    import sys

    sys.path.insert(1, "/home/ubuntu/mlab2_https/mlab2/")
    from remix_d5_acdc_utils import (
        ACDCCorrespondence,
        ACDCExperiment,
        ACDCInterpNode,
    )
    from interp.circuit.projects.gpt2_gen_induction.rust_path_patching import make_arr
    import rust_circuit as rc
    import torch
    from interp.circuit.interop_rust.model_rewrites import To, configure_transformer
    from interp.circuit.interop_rust.module_library import load_model_id
    from interp.tools.data_loading import get_val_seqs
    from interp.tools.indexer import SLICER as S
    from interp.tools.indexer import TORCH_INDEXER as I
    from interp.tools.rrfs import RRFS_DIR
    from torch.testing import assert_close
    import random
    import pickle
    import remix_d3_test as tests

    MAIN = __name__ == "__main__"

    try:
        with open(
            os.path.expanduser("~/induction/data/masks/mask_repeat_candidates.pkl"),
            "rb",
        ) as f:
            mask_repeat_candidates = pickle.load(f)
            # t[1] has 132, 136, 176 available...
            # if induction is AB...AB, B tokens are on 133(OK...), 137, 177
            # OK so this is where we punish losses
    except:
        raise Exception(
            "Have you cloned https://github.com/aryamanarora/induction ??? It is where all the masks are kept !!!"
        )

    DEVICE = "cuda:0"
    SEQ_LEN = 300
    NUM_EXAMPLES = 40
    MODEL_ID = "attention_only_2"
    PRINT_CIRCUITS = True
    ACTUALLY_RUN = True
    SLOW_EXPERIMENTS = True
    DEFAULT_CHECKS: ExperimentCheck = True
    EVAL_DEVICE = "cuda:0"
    MAX_MEMORY = 20000000000
    # BATCH_SIZE = 2000
    USING_WANDB = True
    MONOTONE_METRIC = "maximize"
    START_TIME = datetime.datetime.now().strftime("%a-%d%b_%H%M%S")
    PROJECT_NAME = f"induction_arthur"

    (circ_dict, tokenizer, model_info) = load_model_id(MODEL_ID)

    """
    Get toks and data
    """

    # longer seq len is better, but short makes stuff a bit easier...
    n_files = 1
    reload_dataset = False
    toks_int_values: rc.Array

    if reload_dataset:
        dataset_toks = torch.tensor(
            get_val_seqs(n_files=n_files, files_start=0, max_size=SEQ_LEN + 1)
        ).cuda()
        NUM_EXAMPLES, _ = dataset_toks.shape
        toks_int_values = rc.Array(dataset_toks.float(), name="toks_int_vals")
        print(f'new dataset "{toks_int_values.repr()}"')
    else:
        P = rc.Parser()
        toks_int_values_raw = P(
            f"'toks_int_vals' [104091,301] Array 3f36c4ca661798003df14994"
        ).cast_array()

    CACHE_DIR = f"{RRFS_DIR}/ryan/induction_scrub/cached_vals"
    good_induction_candidate = torch.load(
        f"{CACHE_DIR}/induction_candidates_2022-10-15 04:48:29.970735.pt"
    ).to(device=DEVICE, dtype=torch.float32)
    assert (
        toks_int_values_raw.shape[0] >= SEQ_LEN
    ), f"toks_int_values_raw.shape[0] = {toks_int_values_raw.shape[0]} < {SEQ_LEN} - you could try increasing `n_files`"
    assert (
        toks_int_values_raw.shape[1] >= SEQ_LEN + 1
    ), f"toks_int_values_raw.shape[1] = {toks_int_values_raw.shape[1]} < {SEQ_LEN + 1}"

    tokens_device_dtype = rc.TorchDeviceDtype(device="cuda", dtype="int64")

    toks_int_values = make_arr(
        toks_int_values_raw.value[:NUM_EXAMPLES, :SEQ_LEN],
        name="toks_int_vals",
        device_dtype=tokens_device_dtype,
    )
    mask_reshaped = mask_repeat_candidates[
        :NUM_EXAMPLES, :SEQ_LEN
    ]  # only used for loss
    denom = mask_reshaped.int().sum().item()
    print("We're going to study", denom, "examples...")
    assert denom == 172, (denom, "was not expected")

    toks_int_labels = make_arr(
        toks_int_values_raw.value[:NUM_EXAMPLES, 1 : SEQ_LEN + 1],
        name="toks_int_labels",
        device_dtype=tokens_device_dtype,
    )

    def shuffle_tensor(tens):
        """Shuffle tensor along first dimension"""
        return tens[torch.randperm(tens.shape[0])]

    toks_int_values_other = make_arr(
        shuffle_tensor(toks_int_values.value[:NUM_EXAMPLES, : SEQ_LEN + 1]),
        name="toks_int_vals_other",
        device_dtype=tokens_device_dtype,
    )

    toks = tokenizer.batch_decode(
        good_induction_candidate.nonzero().flatten().view(-1, 1)
    )
    maxlen_tok = max((len(tok), tok) for tok in toks)

    circ_dict = {
        s: rc.cast_circuit(c, rc.TorchDeviceDtypeOp(device="cuda"))
        for s, c in circ_dict.items()
    }

    orig_circuit = circ_dict["t.bind_w"]
    tok_embeds = circ_dict["t.w.tok_embeds"]
    pos_embeds = circ_dict["t.w.pos_embeds"]

    default_input = toks_int_values.rename("tokens")
    default_output = toks_int_labels.rename("labels")

    print("\ntokens of input and output")
    print(tokenizer.batch_decode(default_input.evaluate()[0, :10]))
    print(tokenizer.batch_decode(default_output.evaluate()[0, :10]))

    patch_input = toks_int_values_other.rename("tokens")
    # make_arr(
    #     toks_int_values_other[:NUM_EXAMPLES, :SEQ_LEN],
    #     "tokens",
    #     device_dtype=tokens_device_dtype,
    # )
    patch_output = default_output  # oo cares..

    default_ds = Dataset({"tokens": default_input, "labels": default_output})
    patch_ds = Dataset({"tokens": patch_input, "labels": patch_output})

    # from transformer_lens import HookedTransformer
    # et_model = HookedTransformer.from_pretrained("gpt2-medium")

    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer("Hello world")["input_ids"]
    [15496, 995]

    """
    Create metric

    As mentioned in the AF post, the UNSCRUBBED output = 0.179,
    Induction heads scrubbed = 0.24
    """

    def negative_log_probs(dataset: Dataset, logits: torch.Tensor,) -> float:
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

        masked_log_probs = log_probs * mask_reshaped.int()
        result = (-1.0 * float(masked_log_probs.sum())) / denom

        print("Result", result, denom)
        return result

    """
    Create model
    """

    assert (
        model_info.causal_mask
    ), "Should not apply causal mask if the transformer doesn't expect it!"

    # TODO: may be an issue later!
    causal_mask = rc.Array(
        (torch.arange(SEQ_LEN)[:, None] >= torch.arange(SEQ_LEN)[None, :]).to(
            tok_embeds.cast_array().value
        ),
        f"t.a.c.causal_mask",
    )
    assert model_info.pos_enc_type == "shortformer"
    pos_embeds = pos_embeds.index(I[:SEQ_LEN], name="t.w.pos_embeds_idxed")

    tokens_arr = rc.cast_circuit(
        rc.Array(torch.zeros(SEQ_LEN).to(torch.long), name="tokens"),
        device_dtype=tokens_device_dtype.op(),
    )
    idxed_embeds = rc.GeneralFunction.gen_index(
        tok_embeds, tokens_arr, index_dim=0, name="idxed_embeds"
    )

    # CHECK
    model = rc.module_new_bind(
        orig_circuit,
        ("t.input", idxed_embeds),
        ("a.mask", causal_mask),
        ("a.pos_input", pos_embeds),
        name="t.call",
    )

    # model = model_info.bind_to_input(
    #     orig_circuit,
    #     idxed_embeds,
    #     pos_embeds,
    #     causal_mask,
    # )

    # CHECK
    model = model.update(
        "t.bind_w",
        lambda c: configure_transformer(
            c,
            To.ATTN_HEAD_MLP_NORM,
            split_by_head_config="full",
            use_pull_up_head_split=True,
            use_flatten_res=True,
            flatten_components=True,
        ),
    )
    model = model.cast_module().substitute()
    model.print_html()
    model = rc.Index(model, I[0]).rename("model")
    model = rc.conform_all_modules(model)
    model = model.update("t.call", lambda c: c.rename("logits"))
    model = model.update("t.call", lambda c: c.rename("logits_with_bias"))
    model = model.update(
        rc.Regex("[am]\\d(.h\\d)?$"), lambda c: c.rename(c.name + ".inner")
    )
    model = model.update("t.inp_tok_pos", lambda c: c.rename("embeds"))
    model = model.update("t.a.mask", lambda c: c.rename("padding_mask"))
    for l in range(model_info.params.num_layers):
        for h in range(8):
            model = model.update(f"b{l}.a.h{h}", lambda c: c.rename(f"a{l}.h{h}"))
        next = "final" if l == model_info.params.num_layers - 1 else f"a{l + 1}"
        model = model.update(f"b{l}", lambda c: c.rename(f"{next}.input"))

    def create_path_matcher(
        start_node: rc.MatcherIn, path: list[str], max_distance=10
    ) -> rc.IterativeMatcher:
        """
        Creates a matcher that matches a path of nodes, given in a list of names, where the maximum distance between each node on the path is max_distance
        """

        initial_matcher = rc.IterativeMatcher(start_node)
        max_dis_path_matcher = lambda name: rc.restrict(
            rc.Matcher(name), end_depth=max_distance
        )
        chain_matcher = initial_matcher.chain(max_dis_path_matcher(path[0]))
        for i in range(1, len(path)):
            chain_matcher = chain_matcher.chain(max_dis_path_matcher(path[i]))
        return chain_matcher

    q_path = [
        "a.comb_v",
        "a.attn_probs",
        "a.attn_scores",
        "a.attn_scores_raw",
        "a.q",
    ]
    k_path = [
        "a.comb_v",
        "a.attn_probs",
        "a.attn_scores",
        "a.attn_scores_raw",
        "a.k",
    ]
    v_path = ["a.comb_v", "a.v"]
    qkv_paths = {"q": q_path, "k": k_path, "v": v_path}
    attention_head_name = "a{layer}.h{head}"
    qkv_node_name = "a{layer}.h{head}.{qkv}"
    embed_name = "idxed_embeds"
    root_name = "final.input"
    no_layers = 2
    no_heads = 8
    new_circuit = model
    for l in range(no_layers):
        for h in range(no_heads):
            for qkv in ["q", "k", "v"]:
                qkv_matcher = create_path_matcher(f"a{l}.h{h}", qkv_paths[qkv])
                new_circuit = new_circuit.update(
                    qkv_matcher, lambda c: c.rename(f"a{l}.h{h}.{qkv}")
                )

    printer = rc.PrintHtmlOptions(
        shape_only_when_necessary=False,
        traversal=rc.restrict(
            rc.IterativeMatcher(
                "embeds", "padding_mask", "final.norm", rc.Regex("^[am]\\d(.h\\d)?$")
            ),
            term_if_matches=True,
        ),
    )
    new_circuit = rc.substitute_all_modules(new_circuit)
    new_circuit.get_unique("final.input").print_html()
    new_circuit = new_circuit.get_unique("logits")

    """
    Create correspondence
    """

    all_names = (
        [embed_name, root_name]
        + [
            attention_head_name.format(layer=l, head=h)
            for l in range(no_layers)
            for h in range(no_heads)
        ]
        + [
            qkv_node_name.format(layer=l, head=h, qkv=qkv)
            for l in range(no_layers)
            for h in range(no_heads)
            for qkv in ["q", "k", "v"]
        ]
    )
    all_names = set(all_names)
    template_corr = ACDCCorrespondence(all_names=all_names)
    root = ACDCInterpNode(root_name, is_root=True)
    template_corr.add(root)

    all_residual_stream_parents: List[ACDCInterpNode] = []
    all_residual_stream_parents.append(root)
    for layer in tqdm(range(no_layers - 1, -1, -1)):
        qkv_nodes_list = []
        for head in range(no_heads):
            head_node = ACDCInterpNode(
                attention_head_name.format(layer=layer, head=head)
            )
            for node in all_residual_stream_parents:
                node.add_child(head_node)
                head_node.add_parent(node)
            template_corr.add(head_node)
            for qkv in ["q", "k", "v"]:
                qkv_node = ACDCInterpNode(f"a{layer}.h{head}.{qkv}")
                head_node.add_child(qkv_node)
                qkv_node.add_parent(head_node)
                template_corr.add(qkv_node)
                qkv_nodes_list.append(qkv_node)
        all_residual_stream_parents += qkv_nodes_list

    # add embedding node
    embed_node = ACDCInterpNode(embed_name)
    for node in tqdm(all_residual_stream_parents):
        node.add_child(embed_node)
        embed_node.add_parent(node)
    template_corr.add(embed_node)

    template_corr.show()
    template_corr.topologically_sort_corr()
    return template_corr


def compute_no_edges_in_transformer_lens(nodes_to_mask):
    """
    copied from ACDC code
    """
    nodes_to_mask = [
        i.replace("_", ".").replace("layer.", "").replace("head.", "")
        for i in nodes_to_mask
    ]
    nodes_to_mask = add_hierarchy_nodes(nodes_to_mask)
    template_corr = get_ACDC_correspondece()
    count = 0
    for node in template_corr.corr:
        parent_name = convert_node_name(node.name)
        children = [i for i in node.children]
        for child in children:
            child_name = convert_node_name(child.name)
            if parent_name not in nodes_to_mask and child_name not in nodes_to_mask:
                count += 1

    return count


def add_hierarchy_nodes(nodes_to_mask):
    """
    copied from ACDC code
    """
    unique_hierarchy_nodes = set(
        [i.replace(".q", "").replace(".k", "").replace(".v", "") for i in nodes_to_mask]
    )
    for hierarchy_node in unique_hierarchy_nodes:
        if (
            hierarchy_node + ".q" in nodes_to_mask
            and hierarchy_node + ".k" in nodes_to_mask
            and hierarchy_node + ".v" in nodes_to_mask
        ):
            nodes_to_mask.append(hierarchy_node)
    return nodes_to_mask


def convert_node_name(node_name: str) -> str:
    node_name = node_name.replace("a", "")
    node_name = node_name.replace("h", "")
    return node_name



class IDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def t(self):
        return self.arrs["tokens"].value

    def l(self):
        return self.arrs["labels"].value

def get_prompts_dataset():
    """Taken from Aengus code"""
    import prompts_file
    import rust_circuit as rc
    import transformer_lens
    from interp.circuit.projects.gpt2_gen_induction.rust_path_patching import make_arr

    other_device_dtype = rc.TorchDeviceDtype("cuda:0", "float32")
    tokens_device_dtype = rc.TorchDeviceDtype("cuda:0", "int64")

    et_model = transformer_lens.HookedTransformer.from_pretrained(model_name="attn-only-4l", is_masked=False)
    # Stefan Heimersheim
    batch_size=50
    # random.seed(0)

    docstring_ind_prompt_kwargs = dict(
        n_matching_args=3,
        n_def_prefix_args=2,
        n_def_suffix_args=1,
        n_doc_prefix_args=0,
        met_desc_len=3,
        arg_desc_len=2
    )

    prompts = [prompts_file.docstring_induction_prompt_generator("rest", **docstring_ind_prompt_kwargs) for _ in range(batch_size)]
    batched_prompts = prompts_file.BatchedPrompts(prompts=prompts, model=et_model)


    # DATASET SETUP
    N = batched_prompts.clean_tokens.shape[0]
    seq_len = batched_prompts.clean_tokens.shape[1]

    assert (N, seq_len) == (50, 41)

    default_data = make_arr(
        batched_prompts.clean_tokens.cpu(),
        # ioi_dataset.toks.long()[:N, : seq_len - 1],
        "tokens",
        device_dtype=tokens_device_dtype,
    )
    patch_data = make_arr(
        batched_prompts.corrupt_tokens["random_doc"].cpu(),
        # abc_Dataset.toks.long()[:N, : seq_len - 1],
        "tokens",
        device_dtype=tokens_device_dtype,
    )
    default_ds = IDataset({"tokens": default_data})
    patch_ds = IDataset({"tokens": patch_data})
    return default_ds, patch_ds

