import os
import sys

from dataset_utils import get_ds, get_tokenized_datasets, get_ds_with_categories
from tokenizer import get_tokenizer
from get_model import get_model
from trainer import TrainerManager

sys.path.append(os.getcwd())

PRETRAINED_MODEL = 'distilgpt2'
DATASET_PATH = os.getenv('DATASET_PATH')
DATASET_W_CATEG_PATH = os.getenv('DATASET_W_CATEG_PATH')
SAVE_PATH = os.getenv('SAVE_MODEL_PATH')
MAX_LEN = 512


def run_trainer(save_path):
    dataset = get_ds_with_categories(DATASET_W_CATEG_PATH, DATASET_PATH)
    tokenizer = get_tokenizer(PRETRAINED_MODEL)

    tokenized_datasets = get_tokenized_datasets(
        dataset, tokenizer, MAX_LEN,
    )
    model = get_model(PRETRAINED_MODEL)
    model.resize_token_embeddings(len(tokenizer))
    # train
    trainer_manager = TrainerManager(
        save_path,
        model,
        tokenizer,
        tokenized_datasets,
        num_train_epochs=5
    )
    trainer_manager.train()


if __name__ == "__main__":
    run_trainer(SAVE_PATH)
