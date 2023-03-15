from transformers import GPT2LMHeadModel


def get_model(model_checkpoint_or_path) -> 'GPT2LMHeadModel':
    return GPT2LMHeadModel.from_pretrained(model_checkpoint_or_path)
