import time
from typing import TYPE_CHECKING

import numpy as np
from tqdm.auto import tqdm
from torch import no_grad as torch_no_grad
from torch import cuda
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import DataCollatorForSeq2Seq, get_scheduler, TrainingArguments, Trainer

if TYPE_CHECKING:
    from transformers import GPT2LMHeadModel


LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-8


class TrainerManager:

    def __init__(self, output_dir, model: 'GPT2LMHeadModel', tokenizer, tokenized_datasets, num_train_epochs=10):
        self.model = model
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        self.num_train_epochs = num_train_epochs
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, self.model)
        self.optimizer = AdamW(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        self.train_dataloader = DataLoader(
            tokenized_datasets["train"],
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=3,
        )
        self.eval_dataloader = DataLoader(
            tokenized_datasets["validation"], collate_fn=self.data_collator, batch_size=3
        )

        self.accelerator = Accelerator()
        # override model, optim and dataloaders to allow Accelerator to autohandle `device`
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader = \
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.eval_dataloader
            )  # type: GPT2LMHeadModel, AdamW, DataLoader, DataLoader
        len_train_dataloader = len(self.train_dataloader)
        num_update_steps_per_epoch = len_train_dataloader
        self.num_training_steps = num_train_epochs * num_update_steps_per_epoch
        # create scheduler with changing learning rate
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=3,
            num_training_steps=self.num_training_steps,
        )

    def train(self):
        cuda.empty_cache()
        progress_bar = tqdm(range(self.num_training_steps))

        for epoch in range(self.num_train_epochs):
            print('======== Epoch {:} / {:} ========'.format(epoch + 1, self.num_train_epochs))
            print('Training...')

            t0 = time.time()
            total_train_loss = 0
            self.model.train()

            for step, batch in enumerate(self.train_dataloader):
                b_labels = batch['input_ids']
                outputs = self.model(
                    input_ids=b_labels,
                    labels=b_labels,
                    attention_mask=batch['attention_mask']
                )

                loss = outputs[0]
                total_train_loss += loss
                self.accelerator.backward(loss)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.model.zero_grad()

                progress_bar.update(1)

            avg_train_loss = total_train_loss / len(self.train_dataloader)
            training_time = time.time() - t0
            print("Average training loss: {0:.2f}".format(avg_train_loss))
            print("Training epoch took: {:}\n".format(training_time))

            self.eval()

    def eval(self):
        self.model.eval()
        total_eval_loss = 0
        nb_eval_steps = 0
        t0 = time.time()

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        for batch in tqdm(self.eval_dataloader):
            with torch_no_grad():
                b_labels = batch['input_ids']
                outputs = unwrapped_model(
                    input_ids=b_labels,
                    labels=b_labels,
                    attention_mask=batch['attention_mask']
                )
                loss = outputs[0]
                batch_loss = loss.item()
                total_eval_loss += batch_loss

        avg_val_loss = total_eval_loss / len(self.eval_dataloader)

        validation_time = time.time() - t0

        print("Validation Loss: {0:.2f}".format(avg_val_loss))
        print("Validation took: {:}".format(validation_time))

        # Save the model and tokenizer
        self.accelerator.wait_for_everyone()
        unwrapped_model.save_pretrained(self.output_dir, save_function=self.accelerator.save)
        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(self.output_dir)
