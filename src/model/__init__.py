# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from logging import getLogger
import os
import torch

from .transformer import TransformerModel


logger = getLogger()


def check_model_params(params):
    """
    Check models parameters.
    """
    # model dimensions
    assert params.enc_emb_dim % params.n_enc_heads == 0
    assert params.dec_emb_dim % params.n_dec_heads == 0

    # reload a pretrained model
    if params.reload_model != "":
        assert os.path.isfile(params.reload_model)


def build_modules(env, params):
    """
    Build modules.
    """
    modules = {}
    modules["encoder"] = TransformerModel(
        params, env.id2word, is_encoder=True, with_output=False
    )
    modules["decoder"] = TransformerModel(
        params, env.id2word, is_encoder=False, with_output=True
    )

    # reload pretrained modules
    if params.reload_model != "":
        logger.info(f"Reloading modules from {params.reload_model} ...")
        reloaded = torch.load(params.reload_model)
        for k, v in modules.items():
            assert k in reloaded
            if all([k2.startswith("module.") for k2 in reloaded[k].keys()]):
                reloaded[k] = {
                    k2[len("module.") :]: v2 for k2, v2 in reloaded[k].items()
                }
            v.load_state_dict(reloaded[k])

    # log
    logger.debug(f"{len(modules)}")
    
    for k, v in modules.items():
        logger.debug(f"{v}: {v}")
    for k, v in modules.items():
        logger.info(
            f"Number of parameters ({k}): {sum([p.numel() for p in v.parameters() if p.requires_grad])}"
        )

    # cuda
    if not params.cpu and not params.metals:
        for v in modules.values():
            v.cuda()

    if params.metals:
        for v in modules.values():
            v.to(torch.device("mps"))

    return modules
