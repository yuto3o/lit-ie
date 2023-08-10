# -*- coding: utf-8 -*-
import collections
import copy
import json
import pprint
import re
from typing import List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SequenceDataset(Dataset):

    def __init__(self, data):
        self.data_source = data

    def __getitem__(self, i):
        return self.data_source[i]

    def __len__(self):
        return len(self.data_source)


def eval_metrics(y_trues, y_preds, strict=True):
    assert len(y_trues) == len(y_preds), f"y_trues(num={len(y_trues)}) != y_preds(num={len(y_preds)})"

    task2metrics = {
        'entities': collections.defaultdict(Metric),
        'relations': collections.defaultdict(Metric),
        'events': collections.defaultdict(Metric),
        'events_args': collections.defaultdict(Metric),
    }

    for y_true, y_pred in zip(y_trues, y_preds):
        assert y_true['text'] == y_pred['text'], f"{y_true} \n {y_pred}"

        y_pred_ent_label2tuple = collections.defaultdict(set)
        if 'entities' in y_pred:
            for ent in y_pred['entities']:
                y_pred_ent_label2tuple[ent['label']].add(
                    (ent['text'], ent['start'], ent['end']) if strict else ent['text']
                )

        y_pred_rel_label2tuple = collections.defaultdict(set)
        if 'relations' in y_pred:
            for rel in y_pred['relations']:

                if 'label' in rel['subject'] and 'label' in rel['object']:
                    y_pred_rel_label2tuple[f"{rel['subject']['label']}-{rel['label']}-{rel['object']['label']}"].add(
                        (
                            rel['subject']['text'], rel['subject']['start'], rel['subject']['end'],
                            rel['object']['text'], rel['object']['start'], rel['object']['end'],
                        )
                        if strict else (rel['subject']['text'], rel['object']['text'])
                    )
                else:
                    y_pred_rel_label2tuple[rel['label']].add(
                        (
                            rel['subject']['text'], rel['subject']['start'], rel['subject']['end'],
                            rel['object']['text'], rel['object']['start'], rel['object']['end'],
                        )
                        if strict else (rel['subject']['text'], rel['object']['text'])
                    )

        y_pred_eve_label2tuple = collections.defaultdict(set)
        y_pred_arg_label2tuple = collections.defaultdict(set)
        if 'events' in y_pred:
            for eve in y_pred['events']:
                # event level
                if 'label' in eve:
                    y_pred_eve_label2tuple[eve['label']].add(
                        (
                            eve['text'], eve['start'], eve['end'],
                            *sorted(
                                [(arg['text'], arg['label'], arg['start'], arg['end']) for arg in eve['arguments']]),
                        )
                        if strict else (eve['text'], *sorted([(arg['text'], arg['label']) for arg in eve['arguments']]))
                    )
                # argument level
                for arg in eve['arguments']:
                    y_pred_arg_label2tuple[f"{eve['label']}-{arg['label']}"].add(
                        (arg['text'], arg['label'], arg['start'], arg['end'])
                        if strict else (arg['text'], arg['label'])
                    )

        y_true_ent_label2tuple = collections.defaultdict(set)
        if 'entities' in y_true:
            for ent in y_true['entities']:
                y_true_ent_label2tuple[ent['label']].add(
                    (ent['text'], ent['start'], ent['end']) if strict else ent['text']
                )

        y_true_rel_label2tuple = collections.defaultdict(set)
        if 'relations' in y_true:
            for rel in y_true['relations']:

                if 'label' in rel['subject'] and 'label' in rel['object']:
                    y_true_rel_label2tuple[f"{rel['subject']['label']}-{rel['label']}-{rel['object']['label']}"].add(
                        (
                            rel['subject']['text'], rel['subject']['start'], rel['subject']['end'],
                            rel['object']['text'], rel['object']['start'], rel['object']['end'],
                        )
                        if strict else (rel['subject']['text'], rel['object']['text'])
                    )
                else:
                    y_true_rel_label2tuple[rel['label']].add(
                        (
                            rel['subject']['text'], rel['subject']['start'], rel['subject']['end'],
                            rel['object']['text'], rel['object']['start'], rel['object']['end'],
                        )
                        if strict else (rel['subject']['text'], rel['object']['text'])
                    )

        y_true_eve_label2tuple = collections.defaultdict(set)
        y_true_arg_label2tuple = collections.defaultdict(set)
        if 'events' in y_true:
            for eve in y_true['events']:
                if 'label' in eve:
                    y_true_eve_label2tuple[eve['label']].add(
                        (
                            eve['text'], eve['start'], eve['end'],
                            *sorted(
                                [(arg['text'], arg['label'], arg['start'], arg['end']) for arg in eve['arguments']]),
                        )
                        if strict else (eve['text'], *sorted([(arg['text'], arg['label']) for arg in eve['arguments']]))
                    )
                for arg in eve['arguments']:
                    y_true_arg_label2tuple[f"{eve['label']}-{arg['label']}"].add(
                        (arg['text'], arg['label'], arg['start'], arg['end'])
                        if strict else (arg['text'], arg['label'])
                    )

        for label in y_true_ent_label2tuple.keys() | y_pred_ent_label2tuple.keys():
            task2metrics['entities'][label].count_instance_f1(
                y_true_ent_label2tuple[label],
                y_pred_ent_label2tuple[label]
            )

        for label in y_true_rel_label2tuple.keys() | y_pred_rel_label2tuple.keys():
            task2metrics['relations'][label].count_instance_f1(
                y_true_rel_label2tuple[label],
                y_pred_rel_label2tuple[label]
            )

        for label in y_true_eve_label2tuple.keys() | y_pred_eve_label2tuple.keys():
            task2metrics['events'][label].count_instance_f1(
                y_true_eve_label2tuple[label],
                y_pred_eve_label2tuple[label]
            )

        for label in y_true_arg_label2tuple.keys() | y_pred_arg_label2tuple.keys():
            task2metrics['events_args'][label].count_instance_f1(
                y_true_arg_label2tuple[label],
                y_pred_arg_label2tuple[label]
            )

    metrics = {}
    micro_metric = Metric()
    for label, metric in sorted(task2metrics['entities'].items()):
        p, r, f1 = metric.compute_f1()
        metrics[f"ent_{label}_p"] = p
        metrics[f"ent_{label}_r"] = r
        metrics[f"ent_{label}_f1"] = f1

        micro_metric.tp += metric.tp
        micro_metric.pred_num += metric.pred_num
        micro_metric.gold_num += metric.gold_num

    if len(task2metrics['entities']):
        macro_p = np.mean([metrics[f"ent_{label}_p"] for label in task2metrics['entities']])
        macro_r = np.mean([metrics[f"ent_{label}_r"] for label in task2metrics['entities']])
        macro_f1 = safe_div(2 * macro_p * macro_r, macro_r + macro_p)
        metrics['ent_macro_p'] = macro_p
        metrics['ent_macro_r'] = macro_r
        metrics['ent_macro_f1'] = macro_f1

        micro_p, micro_r, micro_f1 = micro_metric.compute_f1()
        metrics['ent_micro_p'] = micro_p
        metrics['ent_micro_r'] = micro_r
        metrics['ent_micro_f1'] = micro_f1

    micro_metric = Metric()
    for label, metric in sorted(task2metrics['relations'].items()):
        p, r, f1 = metric.compute_f1()
        metrics[f"rel_{label}_p"] = p
        metrics[f"rel_{label}_r"] = r
        metrics[f"rel_{label}_f1"] = f1

        micro_metric.tp += metric.tp
        micro_metric.pred_num += metric.pred_num
        micro_metric.gold_num += metric.gold_num

    if len(task2metrics['relations']):
        macro_p = np.mean([metrics[f"rel_{label}_p"] for label in task2metrics['relations']])
        macro_r = np.mean([metrics[f"rel_{label}_r"] for label in task2metrics['relations']])
        macro_f1 = safe_div(2 * macro_p * macro_r, macro_r + macro_p)
        metrics['rel_macro_p'] = macro_p
        metrics['rel_macro_r'] = macro_r
        metrics['rel_macro_f1'] = macro_f1

        micro_p, micro_r, micro_f1 = micro_metric.compute_f1()
        metrics['rel_micro_p'] = micro_p
        metrics['rel_micro_r'] = micro_r
        metrics['rel_micro_f1'] = micro_f1

    micro_metric = Metric()
    for label, metric in sorted(task2metrics['events'].items()):
        p, r, f1 = metric.compute_f1()
        metrics[f"eve_{label}_p"] = p
        metrics[f"eve_{label}_r"] = r
        metrics[f"eve_{label}_f1"] = f1

        micro_metric.tp += metric.tp
        micro_metric.pred_num += metric.pred_num
        micro_metric.gold_num += metric.gold_num

    if len(task2metrics['events']):
        macro_p = np.mean([metrics[f"eve_{label}_p"] for label in task2metrics['events']])
        macro_r = np.mean([metrics[f"eve_{label}_r"] for label in task2metrics['events']])
        macro_f1 = safe_div(2 * macro_p * macro_r, macro_r + macro_p)
        metrics['eve_macro_p'] = macro_p
        metrics['eve_macro_r'] = macro_r
        metrics['eve_macro_f1'] = macro_f1

        micro_p, micro_r, micro_f1 = micro_metric.compute_f1()
        metrics['eve_micro_p'] = micro_p
        metrics['eve_micro_r'] = micro_r
        metrics['eve_micro_f1'] = micro_f1

    micro_metric = Metric()
    for label, metric in sorted(task2metrics['events_args'].items()):
        p, r, f1 = metric.compute_f1()
        metrics[f"eve_arg_{label}_p"] = p
        metrics[f"eve_arg_{label}_r"] = r
        metrics[f"eve_arg_{label}_f1"] = f1

        micro_metric.tp += metric.tp
        micro_metric.pred_num += metric.pred_num
        micro_metric.gold_num += metric.gold_num

    if len(task2metrics['events_args']):
        macro_p = np.mean([metrics[f"eve_arg_{label}_p"] for label in task2metrics['events_args']])
        macro_r = np.mean([metrics[f"eve_arg_{label}_r"] for label in task2metrics['events_args']])
        macro_f1 = safe_div(2 * macro_p * macro_r, macro_r + macro_p)
        metrics['eve_arg_macro_p'] = macro_p
        metrics['eve_arg_macro_r'] = macro_r
        metrics['eve_arg_macro_f1'] = macro_f1

        micro_p, micro_r, micro_f1 = micro_metric.compute_f1()
        metrics['eve_arg_micro_p'] = micro_p
        metrics['eve_arg_micro_r'] = micro_r
        metrics['eve_arg_micro_f1'] = micro_f1

    metrics = {k: round(v, 4) for k, v in metrics.items()}

    return metrics


class Metric:
    def __init__(self):
        self.tp = 0.
        self.gold_num = 0.
        self.pred_num = 0.

    def compute_f1(self):
        tp = self.tp
        pred_num = self.pred_num
        gold_num = self.gold_num
        p, r = safe_div(tp, pred_num), safe_div(tp, gold_num)
        f1 = safe_div(2 * p * r, p + r)
        return p, r, f1

    def count_instance_f1(self, gold_set, pred_set):
        self.gold_num += len(gold_set)
        self.pred_num += len(pred_set)

        # dup_gold_list = copy.deepcopy(gold_list)
        # for pred in pred_list:
        #     if pred in dup_gold_list:
        #         self.tp += 1
        #         dup_gold_list.remove(pred)

        self.tp += len(set(gold_set) & set(pred_set))


def safe_div(a, b):
    if b == 0.:
        return 0.
    else:
        return a / b


def pad_sequence(data: torch.tensor, max_length: int = None, dim: int = -1, padding_value: int = -100):
    dim = data[0].ndim + dim if dim < 0 else dim - 1
    if max_length is None:
        max_length = 1
        for _ in data:
            max_length = max(_.size(dim), max_length)

    output = []
    for _ in data:
        pad = []
        for i in range(_.ndim, 0, -1):
            if i - 1 == dim:
                pad += [0, max_length - _.size(i - 1)]
            else:
                pad += [0, 0]

        output.append(F.pad(_, pad, 'constant', padding_value))

    return torch.stack(output, dim=0)


def load_json(file_name: str):
    with open(file_name, 'r', encoding='utf-8') as file:
        return json.load(file)


def dump_json(file_name: str, obj: Dict):
    with open(file_name, 'w', encoding='utf-8') as file:
        file.write(json.dumps(obj, ensure_ascii=False))


def load_jsonl(file_name: str):
    records = []
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            records.append(json.loads(line))
    return records


def dump_jsonl(file_name: str, obj: List[Dict]):
    with open(file_name, 'w', encoding='utf-8') as file:
        for _ in obj:
            s = json.dumps(_, ensure_ascii=False)
            file.write(s + '\n')


def truncate_data(data, tokenizer, max_length):
    examples = []
    for example in data:
        if 'offset' not in example:
            example['offset'] = 0

        if 'id' not in example:
            example['id'] = example['text']

        queue = [example]
        while queue:
            example = queue.pop()

            if len(tokenizer(example['text']).input_ids) > max_length:
                record_0, record_1 = truncate_example(example)

                queue.append(record_0)
                queue.append(record_1)
            else:
                examples.append(example)

    return examples


def truncate_example(example: Dict, correct: bool = True):
    text = example['text']
    diff = [len(text)]

    text = re.sub('(\s)(\S)', r"\1\n\2", text)
    text = re.sub('([。！？\?!])([^”’)\]）】])', r"\1\n\2", text)
    text = re.sub('(\.{3,})([^”’)\]）】….])', r"\1\n\2", text)
    text = re.sub('(\…+)([^”’)\]）】….])', r"\1\n\2", text)
    text = re.sub('([。！？\?!]|\.{3,}|\…+)([”’)\]）】])([^，。！？\?….])', r'\1\2\n\3', text)
    texts = text.split('\n')

    for i in range(1, len(texts)):
        diff.append(
            abs(len(''.join(texts[:i])) - len(''.join(texts[i:])))
        )

    if len(diff) == 1:
        text_0, text_1 = text[:len(text) // 2], text[len(text) // 2:]
    else:
        # select possible split point
        i = np.argmin(diff)

        while correct:
            pop = False
            idx = len(''.join(texts[:i]))

            if 'entities' in example and example['entities']:
                for ent in example['entities']:
                    if ent['start'] < idx <= ent['end']:
                        pop = True
                        break

            if 'relations' in example and example['relations']:
                for rel in example['relations']:
                    if (
                            min(rel['subject']['start'], rel['object']['start'])
                            < idx <=
                            max(rel['subject']['end'], rel['object']['end'])
                    ):
                        pop = True
                        break

            if 'events' in example and example['events']:
                for eve in example['events']:
                    if eve['start'] < idx <= eve['end'] or (
                            eve['arguments'] and
                            min([arg['start'] for arg in eve['arguments'] if arg['start'] is not None])
                            < idx <=
                            max([arg['end'] for arg in eve['arguments'] if arg['end'] is not None])
                    ):
                        pop = True
                        break

            if pop:
                diff.pop(int(i))

                if len(diff) == 0:
                    i = len(texts) // 2
                    break

                i = np.argmin(diff)
            else:
                break

        text_0, text_1 = ''.join(texts[:i]), ''.join(texts[i:])

        if not text_0 or not text_1:
            text_0, text_1 = ''.join(texts[:len(texts) // 2]), ''.join(texts[len(texts) // 2:])

    offset = len(text_0)
    record_0 = {'id': example['id'], 'text': text_0, 'offset': example['offset']}
    record_1 = {'id': example['id'], 'text': text_1, 'offset': offset}

    if correct:
        if 'entities' in example:
            record_0['entities'] = []
            record_1['entities'] = []

            for ent in example['entities']:
                if ent['end'] < offset:
                    record_0['entities'].append(
                        copy.deepcopy(ent)
                    )
                elif ent['start'] >= offset:
                    record_1['entities'].append(
                        {
                            'start': ent['start'] - offset,
                            'end': ent['end'] - offset,
                            'text': ent['text'],
                            'label': ent['label']
                        }
                    )

        if 'relations' in example and correct:

            record_0['relations'] = []
            record_1['relations'] = []

            for rel in example['relations']:
                if rel['subject']['end'] < offset and rel['object']['end'] < offset:
                    record_0['relations'].append(copy.deepcopy(rel))

                elif rel['subject']['start'] >= offset and rel['object']['start'] >= offset:
                    record_1['relations'].append(
                        {
                            'subject':
                                {
                                    'start': rel['subject']['start'] - offset,
                                    'end': rel['subject']['end'] - offset,
                                    'text': rel['subject']['text'],
                                    'label': rel['subject']['label'] if 'label' in rel['subject'] else None
                                },

                            'object':
                                {
                                    'start': rel['object']['start'] - offset,
                                    'end': rel['object']['end'] - offset,
                                    'text': rel['object']['text'],
                                    'label': rel['object']['label'] if 'label' in rel['object'] else None
                                },
                            'label': rel['label']
                        }
                    )

        if 'events' in example and correct:

            record_0['events'] = []
            record_1['events'] = []

            for eve in example['events']:
                if max([eve['end']] + [arg['end'] for arg in eve['arguments'] if arg['end']]) < offset:
                    record_0['events'].append(copy.deepcopy(eve))

                elif min([eve['start']] + [arg['start'] for arg in eve['arguments'] if arg['start']]) >= offset:
                    record_1['events'].append(
                        {
                            'label': eve['label'],
                            'text': eve['text'],
                            'start': eve['start'] - offset,
                            'end': eve['end'] - offset,
                            'arguments': [
                                {
                                    'label': arg['label'],
                                    'text': arg['text'],
                                    'start': arg['start'] - offset,
                                    'end': arg['end'] - offset,
                                }
                                for arg in eve['arguments']
                            ]
                        }
                    )

    else:

        if 'entities' in example:
            record_0['entities'] = example['entities']
            record_1['entities'] = example['entities']

        if 'relations' in example:
            record_0['relations'] = example['relations']
            record_1['relations'] = example['relations']

        if 'events' in example:
            record_0['events'] = example['events']
            record_1['events'] = example['events']

    return record_0, record_1


def print_average_token_length(data, tokenizer, logger=None):
    lens = []
    for example in data:
        lens.append(len(tokenizer(example['text']).input_ids))

    msg = f"Average Length of Tokens(25%,50%,75%,95%): {np.percentile(lens, [25, 50, 75, 95])}"

    if logger:
        logger.info(msg)
    else:
        print(msg)


def print_summary_data(data, logger=None):
    ent2cnt, rel2cnt, eve2cnt = collections.defaultdict(int), collections.defaultdict(int), collections.defaultdict(int)
    for record in data:

        if 'entities' in record:
            for ent in record['entities']:
                ent2cnt[ent['label']] += 1

        if 'relations' in record:
            for rel in record['relations']:
                if 'label' in rel['subject'] and 'label' in rel['object']:
                    rel2cnt[(rel['subject']['label'], rel['label'], rel['object']['label'])] += 1
                else:
                    rel2cnt[rel['label']] += 1

        if 'events' in record:
            for evt in record['events']:
                eve2cnt[evt['label']] += 1
                for arg in evt['arguments']:
                    eve2cnt[(evt['label'], arg['label'])] += 1

    msg = pprint.pformat({'total': len(data), **ent2cnt, **rel2cnt, **eve2cnt})

    if logger:
        logger.info(msg)
    else:
        print(msg)
