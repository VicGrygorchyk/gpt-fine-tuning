from typing import TYPE_CHECKING, Dict, List
from glob import glob
import json

import pyarrow as pa
import torch
from datasets import DatasetDict, Dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def get_ds(data_path) -> 'DatasetDict':
    tables = []
    for data_file in glob(f'{data_path}/*'):
        with open(data_file, 'r+') as f:
            json_f = json.load(f)  # type: Dict['text']
            tables.extend([{'text': val['text']} for key, val in json_f.items()])

    arrow_arr = pa.array(tables)
    schema = pa.schema([
        pa.field('text', pa.string())
    ])

    # with pa.OSFile(f'${data_path}/feedbacks.arrow', 'wb') as sink:
    #     with pa.ipc.new_file(sink, schema=schema) as writer:
    #         batch = pa.record_batch([arrow_arr], schema=schema)
    #         writer.write(batch)
    raw_dataset = Dataset.from_list(tables)

    split_datasets = raw_dataset.train_test_split(train_size=0.95, seed=39)
    split_datasets['validation'] = split_datasets.pop('test')
    return split_datasets


def get_ds2(data_path) -> List:
    tables = []
    with open(data_path, 'r+') as f:
        json_f = json.load(f)  # type: List[Dict['text', 'category']]
        tables = [{'text': f'{item["category"]} {item["text"]}'} for item in json_f]

    return tables


def get_ds_with_categories(data_path, data_path2) -> 'DatasetDict':
    def clean_dataset(value_dict):
        results = []
        for key, val in value_dict.items():
            msg: str = val["msg"]
            if len(msg) < 20:
                continue
            if msg.lower().endswith(
                    ("добрий день", "доброго дня", "доброго вечора", "добрый день", "здравствуйте",
                     "добрый вечер",
                     "добрий вечір", "доброго ранку", "добрий ранок", "вітання", "вітаю")
            ):
                continue
            if msg.lower() in ("дякую", "спасибо"):
                continue
            results.append({'text': f'{val["product"]} {val["msg"]}'})
        return results

    tables = []
    for data_file in glob(f'{data_path}/*'):
        with open(data_file, 'r+') as f:
            json_f = json.load(f)  # type: Dict['text']
            tables.extend(clean_dataset(json_f))
    tables.extend(get_ds2(data_path2))
    # arrow_arr = pa.array(tables)
    # schema = pa.schema([
    #     pa.field('text', pa.string()),
    #     pa.field('product', pa.string()),
    #     pa.field('service', pa.string())
    # ])
    #
    # with pa.OSFile(f'${data_path}/feedbacks_w_categories.arrow', 'wb') as sink:
    #     with pa.ipc.new_file(sink, schema=schema) as writer:
    #         batch = pa.record_batch([arrow_arr], schema=schema)
    #         writer.write(batch)
    raw_dataset = Dataset.from_list(tables)

    split_datasets = raw_dataset.train_test_split(train_size=0.95, seed=39)
    split_datasets['validation'] = split_datasets.pop('test')
    return split_datasets


def get_tokenized_datasets(
        datasets: 'DatasetDict',
        tokenizer: 'PreTrainedTokenizerBase',
        max_length: int) -> 'DatasetDict':

    def preprocess_function(examples):
        inputs = ['<|startoftext|>' + ex + '<|endoftext|>' for ex in examples['text']]
        model_inputs = tokenizer(inputs, truncation=True, max_length=max_length, padding="max_length")
        return model_inputs

    tokenized_datasets = datasets.map(preprocess_function, batched=True, remove_columns=datasets['train'].column_names)
    return tokenized_datasets
