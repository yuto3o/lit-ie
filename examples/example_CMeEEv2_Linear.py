# -*- coding: utf-8 -*-
import logging
import os
import sys

import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI

sys.path.append('.')

from lit_ie.utils import load_json, print_average_token_length, truncate_data, print_summary_data, eval_metrics
from lit_ie.lit_module import LitModule, LitDataModule

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s: %(asctime)s] %(message)s',
    datefmt='%m-%d %I:%M:%S',
)
logger = logging.getLogger(__name__)

NUM_CPU_DEVICES = os.cpu_count() // 2

schema2id = {
    'dis': 0,
    'sym': 1,
    'pro': 2,
    'equ': 3,
    'dru': 4,
    'ite': 5,
    'bod': 6,
    'dep': 7,
    'mic': 8,
    'others': 9,
}

id2schema = {k: v for v, k in schema2id.items()}


def decode_bio(labels):
    seq_decode = []
    chunk = [-1, -1, -1]

    for i, token_i in enumerate(labels):

        if token_i % 2 == 0 and token_i != schema2id['others'] * 2:

            # early stop
            if chunk[2] != -1:
                seq_decode.append(chunk)

            chunk = [-1, -1, -1]
            chunk[0] = id2schema[token_i // 2]
            chunk[1] = i
            chunk[2] = i

            if i == len(labels) - 1 and chunk[2] != -1:
                seq_decode.append(chunk)

        elif token_i % 2 == 1 and chunk[1] != -1 and token_i != schema2id['others'] * 2:
            t = id2schema[token_i // 2]

            if t == chunk[0]:
                chunk[2] = i

            if i == len(labels) - 1 and chunk[2] != -1:
                seq_decode.append(chunk)

        else:
            if chunk[2] != -1:
                seq_decode.append(chunk)

            chunk = [-1, -1, -1]

    return seq_decode


class LitCLI(LightningCLI):

    def add_arguments_to_parser(self, parser):
        parser.link_arguments('model.pretrained_model_name_or_path', 'data.pretrained_model_name_or_path')

        parser.add_lightning_class_args(ModelCheckpoint, 'model_checkpoint')

        parser.set_defaults(
            {
                'model_checkpoint.filename': '{epoch:02d}-{step:06d}-{train_loss:.4f}-{val_loss:.4f}-{ent_macro_f1:.4f}-{ent_micro_f1:.4f}',
                'model_checkpoint.monitor': 'ent_micro_f1',
                'model_checkpoint.mode': 'max',
                'model_checkpoint.save_top_k': 1,
            }
        )


class DataModule(LitDataModule):

    def __init__(
            self,
            pretrained_model_name_or_path: str,

            train_data_path: str = None,
            val_data_path: str = None,
            test_data_path: str = None,

            num_workers: int = NUM_CPU_DEVICES,
            train_batch_size: int = 8,
            eval_batch_size: int = 8,

            max_length: int = 2048,
            label_pad_token_id: int = -100,
    ):
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            test_data_path=test_data_path,
            num_workers=num_workers,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            max_length=max_length,
            label_pad_token_id=label_pad_token_id,
        )
        self.save_hyperparameters()

    def process_data(self, data):

        examples = []
        for record in load_json(data):
            example = {'text': record['text'], 'entities': []}

            for ent in record['entities']:
                example['entities'].append(
                    {'label': ent['type'], 'text': ent['entity'], 'start': ent['start_idx'], 'end': ent['end_idx'] - 1}
                )
            examples.append(example)

        print_average_token_length(examples, self.tokenizer, logger)
        examples = truncate_data(examples, self.tokenizer, self.hparams.max_length)
        print_average_token_length(examples, self.tokenizer, logger)
        print_summary_data(examples, logger)

        for example in examples:

            tokenized_inputs = self.tokenizer(example['text'], return_offsets_mapping=True)
            input_ids = tokenized_inputs.input_ids[:self.hparams.max_length]
            offset_mapping = tokenized_inputs.offset_mapping[:self.hparams.max_length]

            example['input_ids'] = input_ids
            example['offset_mapping'] = offset_mapping

            start_mapping = {j[0]: i for i, j in enumerate(offset_mapping) if j[0] != j[1]}
            end_mapping = {j[1] - 1: i for i, j in enumerate(offset_mapping) if j[0] != j[1]}

            def align(start, end):
                return start_mapping[start], end_mapping[end]

            labels = [schema2id['others'] * 2] * len(offset_mapping)

            for ent in example['entities']:
                try:
                    l_diff = len(ent['text']) - len(ent['text'].lstrip())
                    r_diff = len(ent['text']) - len(ent['text'].rstrip())

                    start, end = align(ent['start'] + l_diff, ent['end'] - r_diff)
                except KeyError:
                    continue

                labels[start] = schema2id[ent['label']] * 2
                labels[start + 1:end + 1] = [schema2id[ent['label']] * 2 + 1] * (end - start)

            example['labels'] = torch.LongTensor(labels)

        return examples


class Module(LitModule):

    def __init__(
            self,
            pretrained_model_name_or_path: str,

            warmup_ratio: float = 0.05,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-5,
            bert_learning_rate: float = 2e-5,
            bert_weight_decay: float = 0.01,

            out_file_path: str = 'result.jsonl'
    ):
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            warmup_ratio=warmup_ratio,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            bert_learning_rate=bert_learning_rate,
            bert_weight_decay=bert_weight_decay,
            out_file_path=out_file_path,
        )
        self.save_hyperparameters()

        config = self.model.config
        self.linear = torch.nn.Linear(config.hidden_size, len(schema2id) * 2 - 1)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(
            self,
            input_ids,
            attention_mask,
            token_type_ids=None,
            labels=None
    ):

        hidden_state = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ).last_hidden_state

        logits = self.linear(hidden_state)

        output = {
            'logits': logits,
            'loss': None
        }

        if labels is not None:
            output['loss'] = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        return output

    def decode(self, examples, outputs):

        outputs['logits'] = outputs['logits'].to(torch.float32).numpy(force=True)

        records = []
        for b, example in enumerate(examples):

            offset_mapping = example['offset_mapping']
            text = example['text']

            record = {
                'text': text,
                'entities': [],
                'offset': example['offset']
            }

            logits = outputs['logits'][b]
            labels = np.argmax(logits, axis=-1)

            for label, start, end in decode_bio(labels):
                if len(offset_mapping) <= start or len(offset_mapping) <= end:
                    continue

                start, end = offset_mapping[start][0], offset_mapping[end][1]
                record['entities'].append(
                    {
                        'text': text[start:end],
                        'label': label,
                        'start': start,
                        'end': end - 1,
                    }
                )

            records.append(record)

        return records

    def evaluate(self, y_trues, y_preds):
        return eval_metrics(y_trues, y_preds)


def main():
    _ = LitCLI(Module, DataModule, seed_everything_default=42)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    main()
