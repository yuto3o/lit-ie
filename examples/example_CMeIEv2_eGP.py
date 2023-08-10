# -*- coding: utf-8 -*-
import logging
import os
import sys

import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI

sys.path.append('.')

from lit_ie.utils import pad_sequence, load_json, load_jsonl, eval_metrics, print_average_token_length, \
    print_summary_data, truncate_data
from lit_ie.lit_module import LitModule, LitDataModule
from lit_ie.module import EfficientGlobalPointer, SparseMultiLabelCELossWithLogitsLoss

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s: %(asctime)s] %(message)s',
    datefmt='%m-%d %I:%M:%S',
)
logger = logging.getLogger(__name__)

NUM_CPU_DEVICES = os.cpu_count() // 2


def read_schema(file_name):
    schema2id, id2schema = {}, {}
    for i, record in enumerate(load_json(file_name)):
        schema2id[(record['subject_type'], record['predicate'], record['object_type'])] = i
        id2schema[i] = (record['subject_type'], record['predicate'], record['object_type'])
    return schema2id, id2schema


# 删掉文件中的第一行注释
schema2id, id2schema = read_schema('data/CMeIE-V2/53_schemas.json')


class LitCLI(LightningCLI):

    def add_arguments_to_parser(self, parser):
        parser.link_arguments('model.pretrained_model_name_or_path', 'data.pretrained_model_name_or_path')
        parser.link_arguments('data.max_length', 'model.max_length')

        parser.add_lightning_class_args(ModelCheckpoint, 'model_checkpoint')

        parser.set_defaults(
            {
                'model_checkpoint.filename': '{epoch:02d}-{step:06d}-{train_loss:.4f}-{val_loss:.4f}-{rel_macro_f1:.4f}-{rel_micro_f1:.4f}',
                'model_checkpoint.monitor': 'rel_micro_f1',
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

        def search(pattern, sequence):
            n = len(pattern)
            for i in range(len(sequence)):
                if sequence[i:i + n] == pattern:
                    return i, i + n - 1
            return -1, -1

        examples = []
        for record in load_jsonl(data):
            example = {'text': record['text'], 'relations': []}

            for spo in record['spo_list']:
                sbj_start, sbj_end = search(spo['subject'], example['text'])
                obj_start, obj_end = search(spo['object']['@value'], example['text'])

                if sbj_start == -1 or obj_start == -1:
                    continue

                example['relations'].append(
                    {
                        'label': spo['predicate'],
                        'subject': {
                            'text': spo['subject'],
                            'label': spo['subject_type'],
                            'start': sbj_start,
                            'end': sbj_end,
                        },
                        'object': {
                            'text': spo['object']['@value'],
                            'label': spo['object_type']['@value'],
                            'start': obj_start,
                            'end': obj_end,
                        }
                    }
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

            example['input_ids'] = torch.LongTensor(input_ids)
            example['offset_mapping'] = offset_mapping

            start_mapping = {j[0]: i for i, j in enumerate(offset_mapping) if j[0] != j[1]}
            end_mapping = {j[1] - 1: i for i, j in enumerate(offset_mapping) if j[0] != j[1]}

            def align(start, end):
                return start_mapping[start], end_mapping[end]

            entities = [set() for _ in range(2)]
            head_ids = [set() for _ in range(len(schema2id))]
            tail_ids = [set() for _ in range(len(schema2id))]
            for rel in example['relations']:
                try:
                    l_diff = len(rel['subject']['text']) - len(rel['subject']['text'].lstrip())
                    r_diff = len(rel['subject']['text']) - len(rel['subject']['text'].rstrip())
                    sbj_start, sbj_end = align(rel['subject']['start'] + l_diff, rel['subject']['end'] - r_diff)

                    l_diff = len(rel['object']['text']) - len(rel['object']['text'].lstrip())
                    r_diff = len(rel['object']['text']) - len(rel['object']['text'].rstrip())
                    obj_start, obj_end = align(rel['object']['start'] + l_diff, rel['object']['end'] - r_diff)
                except KeyError:
                    continue

                entities[0].add((sbj_start, sbj_end))
                entities[1].add((obj_start, obj_end))

                head_ids[schema2id[(rel['subject']['label'], rel['label'], rel['object']['label'])]].add(
                    (sbj_start, obj_start)
                )
                tail_ids[schema2id[(rel['subject']['label'], rel['label'], rel['object']['label'])]].add(
                    (sbj_end, obj_end)
                )

            for label in entities + head_ids + tail_ids:
                if not label:
                    label.add((-100, -100))

            example['labels'] = pad_sequence(
                [torch.LongTensor(list(_)) for _ in entities + head_ids + tail_ids], dim=-2
            )

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
        self.entities = EfficientGlobalPointer(
            config.hidden_size, head_hidden_size, 2, mask_tril=True, max_rope_len=max_length
        )
        self.subjects = EfficientGlobalPointer(
            config.hidden_size, head_hidden_size, len(schema2id), max_rope_len=max_length, use_rope=False
        )
        self.objects = EfficientGlobalPointer(
            config.hidden_size, head_hidden_size, len(schema2id), max_rope_len=max_length, use_rope=False
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

        entities_logits = self.entities(hidden_state, attention_mask)
        subjects_logits = self.subjects(hidden_state, attention_mask)
        objects_logits = self.objects(hidden_state, attention_mask)

        output = {
            'entities_logits': entities_logits,
            'subjects_logits': subjects_logits,
            'objects_logits': objects_logits,
            'loss': None
        }

        if labels is not None:
            entities_labels, subjects_labels, objects_labels = torch.split(
                labels, [2, len(schema2id), len(schema2id)], 1
            )

            output['loss'] = self.criterion(entities_logits, entities_labels) \
                             + self.criterion(subjects_logits, subjects_labels) \
                             + self.criterion(objects_logits, objects_labels)

        return output

    def decode(self, examples, outputs):

        outputs['entities_logits'] = outputs['entities_logits'].to(torch.float32).numpy(force=True)
        outputs['subjects_logits'] = outputs['subjects_logits'].to(torch.float32).numpy(force=True)
        outputs['objects_logits'] = outputs['objects_logits'].to(torch.float32).numpy(force=True)

        records = []
        for b, example in enumerate(examples):

            offset_mapping = example['offset_mapping']
            text = example['text']

            record = {
                'text': text,
                'relations': [],
                'offset': example['offset']
            }

            subjects, objects = set(), set()
            for label, start, end in zip(*np.where(outputs['entities_logits'][b] > 0.)):
                if label == 0:
                    subjects.add((start, end))
                else:
                    objects.add((start, end))

            for subject_start, subject_end in subjects:
                for object_start, object_end in objects:

                    p1 = np.where(outputs['subjects_logits'][b, :, subject_start, object_start] > 0)[0]
                    p2 = np.where(outputs['objects_logits'][b, :, subject_end, object_end] > 0)[0]
                    ps = set(p1) & set(p2)

                    if not ps:
                        continue

                    sbj_start, sbj_end = offset_mapping[subject_start][0], offset_mapping[subject_end][1]
                    obj_start, obj_end = offset_mapping[object_start][0], offset_mapping[object_end][1]

                    sbj = text[sbj_start: sbj_end]
                    obj = text[obj_start: obj_end]

                    for p in ps:
                        record['relations'].append(
                            {
                                'label': id2schema[p][1],
                                'subject': {
                                    'text': sbj,
                                    'label': id2schema[p][0],
                                    'start': sbj_start,
                                    'end': sbj_end - 1,
                                },
                                'object': {
                                    'text': obj,
                                    'label': id2schema[p][2],
                                    'start': obj_start,
                                    'end': obj_end - 1,
                                },
                            }
                        )

            records.append(record)

        return records

    def evaluate(self, y_trues, y_preds):
        return eval_metrics(y_trues, y_preds, False)


def main():
    _ = LitCLI(Module, DataModule, seed_everything_default=42)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    main()
