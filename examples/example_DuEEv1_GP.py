# -*- coding: utf-8 -*-
import collections
import logging
import os
import sys
from itertools import groupby

import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI

sys.path.append('.')

from lit_ie.utils import pad_sequence, load_jsonl, eval_metrics, \
    print_average_token_length, truncate_data, print_summary_data, dump_jsonl
from lit_ie.lit_module import LitModule, LitDataModule
from lit_ie.module import GlobalPointer, SparseMultiLabelCELossWithLogitsLoss

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s: %(asctime)s] %(message)s',
    datefmt='%m-%d %I:%M:%S',
)
logger = logging.getLogger(__name__)

NUM_CPU_DEVICES = os.cpu_count() // 2


def read_schema(file_name):
    schemas = []
    for record in load_jsonl(file_name):

        schemas.append((record['event_type'], '触发词'))

        for elem in record['role_list']:
            schemas.append((record['event_type'], elem['role']))

    schema2id = {schema: i for i, schema in enumerate(schemas)}
    id2schema = {i: schema for i, schema in enumerate(schemas)}
    return schema2id, id2schema


schema2id, id2schema = read_schema('data/DuEE1.0/duee_event_schema.json')


class LitCLI(LightningCLI):

    def add_arguments_to_parser(self, parser):
        parser.link_arguments('model.pretrained_model_name_or_path', 'data.pretrained_model_name_or_path')
        parser.link_arguments('data.max_length', 'model.max_length')

        parser.add_lightning_class_args(ModelCheckpoint, 'model_checkpoint')

        parser.set_defaults(
            {
                'model_checkpoint.filename': '{epoch:02d}-{step:06d}-{train_loss:.4f}-{val_loss:.4f}-{eve_macro_f1:.4f}-{eve_micro_f1:.4f}',
                'model_checkpoint.monitor': 'eve_micro_f1',
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
        for record in load_jsonl(data):
            example = {'id': record['id'], 'text': record['text'], 'events': []}

            if 'event_list' in record:

                for eve in record['event_list']:
                    tri_start = eve['trigger_start_index']
                    tri_end = eve['trigger_start_index'] + len(eve['trigger']) - 1

                    event = {
                        'label': eve['event_type'],
                        'text': eve['trigger'],
                        'start': tri_start,
                        'end': tri_end,
                        'arguments': []
                    }

                    for arg in eve['arguments']:
                        arg_start = arg['argument_start_index']
                        arg_end = arg['argument_start_index'] + len(arg['argument']) - 1

                        event['arguments'].append(
                            {
                                'label': arg['role'],
                                'text': arg['argument'],
                                'start': arg_start,
                                'end': arg_end,
                            }
                        )

                    example['events'].append(
                        event
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

            entities = [set() for _ in range(len(schema2id))]
            head_ids = [set()]
            tail_ids = [set()]
            for eve in example['events']:
                try:
                    l_diff = len(eve['text']) - len(eve['text'].lstrip())
                    r_diff = len(eve['text']) - len(eve['text'].rstrip())
                    tri_start, tri_end = align(eve['start'] + l_diff, eve['end'] - r_diff)
                except KeyError:
                    continue

                entities[schema2id[(eve['label'], '触发词')]].add((tri_start, tri_end))

                for i0, arg0 in enumerate([eve] + eve['arguments']):
                    try:
                        l_diff = len(arg0['text']) - len(arg0['text'].lstrip())
                        r_diff = len(arg0['text']) - len(arg0['text'].rstrip())
                        arg0_start, arg0_end = align(arg0['start'] + l_diff, arg0['end'] - r_diff)
                    except KeyError:
                        continue

                    if i0 > 0:
                        # 跳过第一个trigger
                        entities[schema2id[(eve['label'], arg0['label'])]].add((arg0_start, arg0_end))

                    for i1, arg1 in enumerate([eve] + eve['arguments']):

                        if i1 > i0:

                            try:
                                l_diff = len(arg1['text']) - len(arg1['text'].lstrip())
                                r_diff = len(arg1['text']) - len(arg1['text'].rstrip())
                                arg1_start, arg1_end = align(arg1['start'] + l_diff, arg1['end'] - r_diff)
                            except KeyError:
                                continue

                            head_ids[0].add((min(arg0_start, arg1_start), max(arg0_start, arg1_start)))
                            tail_ids[0].add((min(arg0_end, arg1_end), max(arg0_end, arg1_end)))

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
        self.arguments = GlobalPointer(
            config.hidden_size, head_hidden_size, len(schema2id), mask_tril=True, max_rope_len=max_length
        )
        self.heads = GlobalPointer(
            config.hidden_size, head_hidden_size, 1, max_rope_len=max_length, use_rope=False, mask_tril=True
        )
        self.tails = GlobalPointer(
            config.hidden_size, head_hidden_size, 1, max_rope_len=max_length, use_rope=False, mask_tril=True
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

        arguments_logits = self.arguments(hidden_state, attention_mask)
        heads_logits = self.heads(hidden_state, attention_mask)
        tails_logits = self.tails(hidden_state, attention_mask)

        output = {
            'arguments_logits': arguments_logits,
            'heads_logits': heads_logits,
            'tails_logits': tails_logits,
            'loss': None
        }

        if labels is not None:
            arguments_labels, heads_labels, tails_labels = torch.split(
                labels, [len(schema2id), 1, 1], 1
            )

            output['loss'] = self.criterion(arguments_logits, arguments_labels) \
                             + self.criterion(heads_logits, heads_labels) \
                             + self.criterion(tails_logits, tails_labels)

        return output

    def decode(self, examples, outputs):

        outputs['arguments_logits'] = outputs['arguments_logits'].to(torch.float32).numpy(force=True)
        outputs['heads_logits'] = outputs['heads_logits'].to(torch.float32).numpy(force=True)
        outputs['tails_logits'] = outputs['tails_logits'].to(torch.float32).numpy(force=True)

        records = []
        for b, example in enumerate(examples):

            offset_mapping = example['offset_mapping']
            text = example['text']

            record = {
                'id': example['id'],
                'text': text,
                'events': [],
                'offset': example['offset']
            }

            args = set()
            for label, start, end in zip(*np.where(outputs['arguments_logits'][b] > 0.)):
                args.add((*id2schema[label], start, end))

            links = set()
            for i0, (_, _, start0, end0) in enumerate(args):
                for i1, (_, _, start1, end1) in enumerate(args):
                    if i1 > i0:
                        if outputs['heads_logits'][b, 0, min(start0, start1), max(start0, start1)] > 0. \
                                and outputs['tails_logits'][b, 0, min(end0, end1), max(end0, end1)] > 0.:
                            links.add((start0, end0, start1, end1))
                            links.add((start1, end1, start0, end0))

            for _, sub_args in groupby(sorted(args), key=lambda x: x[0]):
                for eve in clique_search(list(sub_args), links):

                    event = {'arguments': []}
                    for event_type, role, start, end in eve:
                        start, end = offset_mapping[start][0], offset_mapping[end][1]
                        arg = text[start: end]
                        event['label'] = event_type

                        if role == '触发词':
                            event['text'] = arg
                            event['start'] = start
                            event['end'] = end - 1
                        else:
                            event['arguments'].append(
                                {'text': arg, 'label': role, 'start': start, 'end': end - 1}
                            )

                    # 无触发词
                    # if 'text' not in event:
                    #     continue

                    if len(event['arguments']) == 0:
                        continue

                    record['events'].append(event)

            records.append(record)

        return records

    def evaluate(self, y_trues, y_preds):
        return eval_metrics(y_trues, y_preds, False)

    def on_test_epoch_end(self):

        records = []
        for output in self.outputs:
            records.append(output['records'])

        self.outputs.clear()
        records = sum(records, start=[])

        id2example = collections.defaultdict(list)
        for record in records:
            id2example[record['id']].extend(
                [
                    {
                        'event_type': event['label'],
                        'arguments': [
                            {'role': arg['label'], 'argument': arg['text']}
                            for arg in event['arguments']
                        ],

                    } for event in record['events']
                ]
            )

        records = [{'id': id, 'event_list': id2example[id]} for id in id2example]
        dump_jsonl(self.hparams.out_file_path, records)


# https://github.com/bojone/GPLinker/blob/main/duee_fin.py
class DedupList(list):

    def append(self, x):
        if x not in self:
            super(DedupList, self).append(x)


def neighbors(host, argus, links):
    results = [host]
    for argu in argus:
        if host[2:] + argu[2:] in links:
            results.append(argu)
    return list(sorted(results))


def clique_search(argus, links):
    Argus = DedupList()
    for i1, (_, _, h1, t1) in enumerate(argus):
        for i2, (_, _, h2, t2) in enumerate(argus):
            if i2 > i1:
                if (h1, t1, h2, t2) not in links:
                    Argus.append(neighbors(argus[i1], argus, links))
                    Argus.append(neighbors(argus[i2], argus, links))
    if Argus:
        results = DedupList()
        for A in Argus:
            for a in clique_search(A, links):
                results.append(a)
        return results
    else:
        return [list(sorted(argus))]


def main():
    _ = LitCLI(Module, DataModule, seed_everything_default=42)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    main()
