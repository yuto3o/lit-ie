# -*- coding: utf-8 -*-
import logging
import pprint

import torch
from lightning import LightningModule, LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup, DataCollatorWithPadding

from lit_ie.utils import dump_jsonl, SequenceDataset, pad_sequence

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s: %(asctime)s] %(message)s',
    datefmt='%m-%d %I:%M:%S',
)
logger = logging.getLogger(__name__)


class LitDataCollator(DataCollatorWithPadding):
    r"""
    Data collator for ChatGLM. It is capable of dynamically padding for batched data.
    """
    HF_INPUTS_KEY = [
        'input_ids', 'attention_mask', 'token_type_ids', 'labels',
    ]

    def __init__(
            self,
            tokenizer,
            label_pad_token_id=-100
    ):
        super().__init__(tokenizer, padding=True)
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features, return_tensors=None):
        r"""
        Pads batched data to the longest sequence in the batch.
        """
        hf_features, examples = [], []
        for feature in features:
            hf_features.append({k: v for k, v in feature.items() if k in self.HF_INPUTS_KEY})
            examples.append({k: v for k, v in feature.items() if k not in self.HF_INPUTS_KEY})

        if return_tensors:
            self.return_tensors = return_tensors

        labels = []
        if 'labels' in hf_features[0]:
            for feature in hf_features:
                labels.append(feature.pop('labels'))

        hf_features_batch = super().__call__(hf_features)

        if labels:
            labels = [torch.LongTensor(label) for label in labels]
            if labels[0].ndim == 1:
                hf_features_batch['labels'] = pad_sequence(labels, padding_value=self.label_pad_token_id)
            else:
                hf_features_batch['labels'] = pad_sequence(labels, dim=-2, padding_value=self.label_pad_token_id)

        return {'hf_inputs': hf_features_batch, 'examples': examples}


class LitDataModule(LightningDataModule):

    def __init__(
            self,
            pretrained_model_name_or_path: str,

            train_data_path: str = None,
            val_data_path: str = None,
            test_data_path: str = None,

            num_workers: int = 1,
            train_batch_size: int = 8,
            eval_batch_size: int = 8,

            max_length: int = 2048,
            label_pad_token_id: int = -100,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            use_fast=True,
        )
        self.tokenizer.deprecation_warnings['Asking-to-pad-a-fast-tokenizer'] = True
        self.data = {'train': [], 'val': [], 'test': []}

        self.data_collator = LitDataCollator(
            tokenizer=self.tokenizer,
            label_pad_token_id=label_pad_token_id,
        )

    def process_data(self, data):
        raise NotImplementedError()

    def prepare_data(self):

        logger.info('Preparing Data ...')

        if self.hparams.train_data_path:
            logger.info(f"Loading Train Data from {self.hparams.train_data_path}")
            self.data['train'] = self.process_data(self.hparams.train_data_path)

        if self.hparams.val_data_path:
            logger.info(f"Loading Val Data from {self.hparams.val_data_path}")
            self.data['val'] = self.process_data(self.hparams.val_data_path)

        if self.hparams.test_data_path:
            logger.info(f"Loading Test Data from {self.hparams.test_data_path}")
            self.data['test'] = self.process_data(self.hparams.test_data_path)

        logger.info({k: len(v) for k, v in self.data.items()})

    def setup(self, stage):

        if stage == 'fit':
            self.train_dataset = SequenceDataset(
                self.data['train'],
            )

        if stage in {'fit', 'validate'}:
            self.val_dataset = SequenceDataset(
                self.data['val'],
            )

        if stage in {'fit', 'test'}:
            self.test_dataset = SequenceDataset(
                self.data['test']
            )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):

        for k in batch['hf_inputs']:
            if isinstance(batch['hf_inputs'][k], torch.Tensor):
                batch['hf_inputs'][k] = batch['hf_inputs'][k].to(device)
            else:
                batch['hf_inputs'][k] = [_.to(device) for _ in batch['hf_inputs'][k]]

        return batch

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.train_batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
            drop_last=True,
            num_workers=self.hparams.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.eval_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
            num_workers=self.hparams.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.eval_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
            num_workers=self.hparams.num_workers)


class LitModule(LightningModule):

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
        super().__init__()
        self.save_hyperparameters()

        model = AutoModel.from_pretrained(
            pretrained_model_name_or_path
        )

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            use_fast=True
        )

        self.model = model
        self.tokenizer = tokenizer
        self.outputs = []

    def configure_optimizers(self):

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        params = [
            {
                'params': [p for n, p in self.named_parameters() if
                           n.startswith('model') and not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.hparams.bert_weight_decay,
                'lr': self.hparams.bert_learning_rate
            },
            {
                'params': [p for n, p in self.named_parameters() if
                           n.startswith('model') and any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0,
                'lr': self.hparams.bert_learning_rate
            },
            {
                'params': [p for n, p in self.named_parameters() if not n.startswith('model') and p.requires_grad],
            }
        ]
        optimizer = torch.optim.AdamW(params, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.hparams.warmup_ratio * self.trainer.estimated_stepping_batches),
            num_training_steps=self.trainer.estimated_stepping_batches
        )

        return [{
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }]

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def decode(self, examples, outputs):
        raise NotImplementedError()

    def evaluate(self, examples, outputs):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        output = self(**batch['hf_inputs'])
        self.log_dict({'loss': output['loss'].item()}, prog_bar=True)
        self.outputs.append(output['loss'].item())
        return output['loss']

    def on_validation_epoch_start(self):
        loss = sum(self.outputs) / (len(self.outputs) + 1e-6)
        self.outputs.clear()
        self.log_dict({'train_loss': loss}, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch['hf_inputs'])
        records = self.decode(batch['examples'], outputs)
        self.outputs.append(
            {
                'loss': outputs['loss'].item(),
                'records': records,
                'examples': batch['examples']
            }
        )

    def on_validation_epoch_end(self):
        loss, records, examples = [], [], []
        for output in self.outputs:
            loss.append(output['loss'])
            records.append(output['records'])
            examples.append(output['examples'])

        self.outputs.clear()
        loss = sum(loss) / (len(loss) + 1e-6)

        examples = sum(examples, start=[])
        records = sum(records, start=[])
        metrics = self.evaluate(examples, records)

        metrics = {
            'val_loss': loss,
            **metrics
        }

        # TIPs: too many elements may slow down the training
        log_metrics = {'val_loss': loss}
        if 'ent_macro_f1' and 'ent_micro_f1' in metrics:
            log_metrics['ent_macro_f1'] = metrics['ent_macro_f1']
            log_metrics['ent_micro_f1'] = metrics['ent_micro_f1']
        if 'rel_macro_f1' and 'rel_micro_f1' in metrics:
            log_metrics['rel_macro_f1'] = metrics['rel_macro_f1']
            log_metrics['rel_micro_f1'] = metrics['rel_micro_f1']
        if 'eve_macro_f1' and 'eve_micro_f1' in metrics:
            log_metrics['eve_macro_f1'] = metrics['eve_macro_f1']
            log_metrics['eve_micro_f1'] = metrics['eve_micro_f1']

        self.log_dict(
            log_metrics,
            prog_bar=True
        )

        logger.info(pprint.pformat(metrics, sort_dicts=False))

    def test_step(self, batch, batch_idx):

        outputs = self(
            input_ids=batch['hf_inputs']['input_ids'],
            attention_mask=batch['hf_inputs']['attention_mask'],
            token_type_ids=batch['hf_inputs'].pop('token_type_ids', None)
        )
        records = self.decode(batch['examples'], outputs)
        self.outputs.append(
            {
                'records': records
            }
        )

    def on_test_epoch_end(self):

        records = []
        for output in self.outputs:
            records.append(output['records'])

        self.outputs.clear()

        records = sum(records, start=[])
        dump_jsonl(self.hparams.out_file_path, records)

    def on_fit_end(self):
        # do test at last
        self.trainer.test(dataloaders=self.trainer.datamodule.test_dataloader(), ckpt_path='best')
