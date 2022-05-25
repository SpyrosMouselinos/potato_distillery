from .utils import validate
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM


def distill_bert(model_version="bert-base-cased", keep_layers=4, verbose=True, perform_check=False):
    """
    Distills a BERT model, keeping only the first <keep_layers> layers.
    :param perform_check: Whether to perform post-loading check (WARNING! USES DOUBLE THE CPU MEM)
    :param model_version: The huggingface model code to user.
    :param keep_layers: The number of Transformer Blocks to keep.
    :param verbose: Whether to show additional info during parameter validation.
    :return: The distilled model.
    """
    original_model = AutoModelForMaskedLM.from_pretrained(model_version)
    config = deepcopy(original_model.config)
    config.num_hidden_layers = keep_layers
    if perform_check:
        model = AutoModelForMaskedLM.from_pretrained(model_version, config=config)
        validate(model=model, original_model=original_model, verbose=verbose, name='BERT')
    else:
        del original_model
        model = AutoModelForMaskedLM.from_pretrained(model_version, config=config)
    return model


def distill_gpt2(model_version="gpt2", keep_layers=4, verbose=True, perform_check=False):
    """
    Distills a GPT-2 model, keeping only the first <keep_layers> layers.
    :param perform_check: Whether to perform post-loading check (WARNING! USES DOUBLE THE CPU MEM)
    :param model_version: The huggingface model code to user.
    :param keep_layers: The number of Transformer Blocks to keep.
    :param verbose: Whether to show additional info during parameter validation.
    :return: The distilled model.
    """
    original_model = AutoModelForCausalLM.from_pretrained(model_version)
    config = deepcopy(original_model.config)
    config.num_hidden_layers = keep_layers
    if perform_check:
        model = AutoModelForMaskedLM.from_pretrained(model_version, config=config)
        validate(model=model, original_model=original_model, verbose=verbose, name='GPT-2')
    else:
        del original_model
        model = AutoModelForMaskedLM.from_pretrained(model_version, config=config)
    return model


def distill_gptneo(model_version="EleutherAI/gpt-neo-125M", keep_layers=4, verbose=True, perform_check=False):
    """
    Distills a GPT-2 model, keeping only the first <keep_layers> layers.
    :param perform_check: Whether to perform post-loading check (WARNING! USES DOUBLE THE CPU MEM)
    :param model_version: The huggingface model code to user.
    :param keep_layers: The number of Transformer Blocks to keep.
    :param verbose: Whether to show additional info during parameter validation.
    :return: The distilled model.
    """
    original_model = AutoModelForCausalLM.from_pretrained(model_version)
    config = original_model.config
    config.num_hidden_layers = keep_layers
    if perform_check:
        model = AutoModelForMaskedLM.from_pretrained(model_version, config=config)
        validate(model=model, original_model=original_model, verbose=verbose, name='GPT-Neo')
    else:
        del original_model
        model = AutoModelForMaskedLM.from_pretrained(model_version, config=config)
    return model


def distill_incoder_small(model_version="facebook/incoder-1B", keep_layers=24, verbose=True, perform_check=False):
    """
    Distills a Incoder-1.5B model, keeping only the first <keep_layers> layers.
    :param perform_check: Whether to perform post-loading check (WARNING! USES DOUBLE THE CPU MEM)
    :param model_version: The huggingface model code to user.
    :param keep_layers: The number of Transformer Blocks to keep.
    :param verbose: Whether to show additional info during parameter validation.
    :return: The distilled model.
    """
    original_model = AutoModelForCausalLM.from_pretrained(model_version)
    config = original_model.config
    config.num_hidden_layers = keep_layers
    if perform_check:
        model = AutoModelForMaskedLM.from_pretrained(model_version, config=config)
        validate(model=model, original_model=original_model, verbose=verbose, name='Incoder-Small')
    else:
        del original_model
        model = AutoModelForMaskedLM.from_pretrained(model_version, config=config)
    return model
