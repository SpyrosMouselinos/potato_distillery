from .utils import validate
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM


def distill_bert(model_version="bert-base-cased", keep_layers=4, verbose=True):
    """
    Distills a BERT model, keeping only the first <keep_layers> layers.
    :param model_version: The huggingface model code to user.
    :param keep_layers: The number of Transformer Blocks to keep.
    :param verbose: Whether to show additional info during parameter validation.
    :return: The distilled model.
    """
    original_model = AutoModelForMaskedLM.from_pretrained(model_version)
    config = original_model.config
    config.num_hidden_layers = keep_layers
    model = AutoModelForMaskedLM.from_pretrained(model_version, config=config)
    validate(model=model, original_model=original_model, verbose=verbose, name='BERT')
    return model


def distill_gpt2(model_version="gpt2", keep_layers=4, verbose=True):
    """
    Distills a GPT-2 model, keeping only the first <keep_layers> layers.
    :param model_version: The huggingface model code to user.
    :param keep_layers: The number of Transformer Blocks to keep.
    :param verbose: Whether to show additional info during parameter validation.
    :return: The distilled model.
    """
    original_model = AutoModelForCausalLM.from_pretrained(model_version)
    config = original_model.config
    config.num_hidden_layers = keep_layers
    model = AutoModelForCausalLM.from_pretrained(model_version, config=config)

    validate(model=model, original_model=original_model, verbose=verbose, name='GPT-2')
    return model


def distill_gptneo(model_version="EleutherAI/gpt-neo-125M", keep_layers=4, verbose=True):
    """
    Distills a GPT-2 model, keeping only the first <keep_layers> layers.
    :param model_version: The huggingface model code to user.
    :param keep_layers: The number of Transformer Blocks to keep.
    :param verbose: Whether to show additional info during parameter validation.
    :return: The distilled model.
    """
    original_model = AutoModelForCausalLM.from_pretrained(model_version)
    config = original_model.config
    config.num_hidden_layers = keep_layers
    model = AutoModelForCausalLM.from_pretrained(model_version, config=config)

    validate(model=model, original_model=original_model, verbose=verbose, name='GPT-Neo')
    return model
