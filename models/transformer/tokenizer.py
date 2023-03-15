from transformers import GPT2TokenizerFast


def get_tokenizer(path_or_model: str) -> 'GPT2TokenizerFast':
    """
    Returns a tokenizer from pretrained model.
    :param path_or_model: path to the model, or the model name from HuggingFace lib
    :return: GPT2TokenizerFast
    """
    return GPT2TokenizerFast.from_pretrained(path_or_model, bos_token='<|startoftext|>',
                                             eos_token='<|endoftext|>', pad_token='<|pad|>')
