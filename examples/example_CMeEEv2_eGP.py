# -*- coding: utf-8 -*-
import logging
import os
import sys

import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI

sys.path.append('.')

from lit_ie.utils import pad_sequence, load_json, print_average_token_length, truncate_data, \
    print_summary_data, eval_metrics
from lit_ie.lit_module import LitModule, LitDataModule
from lit_ie.module import EfficientGlobalPointer, SparseMultiLabelCELossWithLogitsLoss

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
}

id2schema = {k: v for v, k in schema2id.items()}


class LitCLI(LightningCLI):

    def add_arguments_to_parser(self, parser):
        parser.link_arguments('model.pretrained_model_name_or_path', 'data.pretrained_model_name_or_path')
        parser.link_arguments('data.max_length', 'model.max_length')

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

            labels = [set() for _ in range(len(schema2id))]
            for ent in example['entities']:
                try:
                    l_diff = len(ent['text']) - len(ent['text'].lstrip())
                    r_diff = len(ent['text']) - len(ent['text'].rstrip())

                    span = align(ent['start'] + l_diff, ent['end'] - r_diff)
                except KeyError:
                    continue

                labels[schema2id[ent['label']]].add(span)

            for label in labels:
                if not label:
                    label.add((-100, -100))

            example['labels'] = pad_sequence([torch.LongTensor(list(_)) for _ in labels], dim=-2)

        return examples


class Module(LitModule):

    def __init__(
            self,
            pretrained_model_name_or_path: str,
            max_length: int = 512,

            head_hidden_size: int = 64,

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
        self.linear = EfficientGlobalPointer(
            config.hidden_size, head_hidden_size, len(schema2id), mask_tril=True, max_rope_len=max_length
        )
        criterion = SparseMultiLabelCELossWithLogitsLoss()

        def _criterion(input, target):
            B, H, L, L = input.size()
            input = input.reshape(B, H, L * L)

            target = target[..., 0] * L + target[..., 1]
            target[target < 0] = -100

            return criterion(input, target)

        self.criterion = _criterion

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

        logits = self.linear(hidden_state, attention_mask)

        output = {
            'logits': logits,
            'loss': None
        }

        if labels is not None:
            output['loss'] = self.criterion(logits, labels)

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

            for label, start, end in zip(*np.where(outputs['logits'][b] > 0.)):
                start, end = offset_mapping[start][0], offset_mapping[end][1]
                record['entities'].append(
                    {
                        'text': text[start:end],
                        'label': id2schema[label],
                        'start': start,
                        'end': end - 1,
                    }
                )

            records.append(record)

    def evaluate(self, y_trues, y_preds):
        return eval_metrics(y_trues, y_preds)


def main():
    _ = LitCLI(Module, DataModule, seed_everything_default=42)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    main()
