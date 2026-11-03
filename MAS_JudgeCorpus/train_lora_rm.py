#!/usr/bin/env python3
"""Standalone LoRA Bradley-Terry reward-model trainer.

This implementation intentionally does not depend on TRL. It consumes the
compact chosen_text/rejected_text JSONL files produced by build_task_filtered_rm.py
and supports torchrun DDP on multiple GPUs.
"""
from __future__ import annotations
import argparse, os, random
from pathlib import Path
import numpy as np
import torch
import yaml
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          BitsAndBytesConfig, DataCollatorWithPadding, Trainer,
                          TrainerCallback, TrainingArguments, set_seed)

class PairCollator:
    def __init__(self, tokenizer): self.tokenizer=tokenizer
    def __call__(self, features):
        c=[{'input_ids':f['chosen_input_ids'],'attention_mask':f['chosen_attention_mask']} for f in features]
        r=[{'input_ids':f['rejected_input_ids'],'attention_mask':f['rejected_attention_mask']} for f in features]
        cp=self.tokenizer.pad(c, padding=True, return_tensors='pt')
        rp=self.tokenizer.pad(r, padding=True, return_tensors='pt')
        return {'chosen_input_ids':cp['input_ids'],'chosen_attention_mask':cp['attention_mask'],
                'rejected_input_ids':rp['input_ids'],'rejected_attention_mask':rp['attention_mask']}

class PairwiseTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        chosen=model(input_ids=inputs['chosen_input_ids'], attention_mask=inputs['chosen_attention_mask']).logits.view(-1)
        rejected=model(input_ids=inputs['rejected_input_ids'], attention_mask=inputs['rejected_attention_mask']).logits.view(-1)
        loss=-torch.nn.functional.logsigmoid(chosen-rejected).mean()
        if return_outputs:
            return loss, {'chosen_scores':chosen.detach(), 'rejected_scores':rejected.detach()}
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # The preference batch has chosen_input_ids/rejected_input_ids rather than
        # the single input_ids expected by AutoModelForSequenceClassification.
        # Trainer's default evaluation path calls model(**inputs) directly when
        # labels are absent, which drops us into Qwen's "exactly one of
        # input_ids/inputs_embeds" error.  Evaluate through the same pairwise
        # Bradley--Terry computation used for training instead.
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        logits = torch.stack((outputs['chosen_scores'], outputs['rejected_scores']), dim=-1)
        return loss.detach(), logits.detach(), None

class EpochSaveCallback(TrainerCallback):
    def __init__(self, directory, tokenizer): self.directory=directory; self.tokenizer=tokenizer
    def on_epoch_end(self,args,state,control,model=None,**kwargs):
        if model is not None and state.epoch is not None and state.is_world_process_zero:
            d=Path(self.directory)/f'epoch-{int(state.epoch)}'; d.mkdir(parents=True,exist_ok=True)
            model.save_pretrained(d); self.tokenizer.save_pretrained(d)

def load_cfg(path):
    with open(path,encoding='utf8') as f: return yaml.safe_load(f)

def build(cfg):
    m=cfg['model']; t=cfg['train']; l=cfg.get('lora',{}); d=cfg['data']
    max_len=int(m.get('max_length',4096)); base=m['base_model']
    tok=AutoTokenizer.from_pretrained(base,local_files_only=True,use_fast=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    tok.model_max_length=max_len
    ds=load_dataset('json',data_files={'train':d['train_file'],'validation':d['val_file']})
    def enc(ex):
        c=tok(ex['chosen_text'],truncation=True,max_length=max_len,add_special_tokens=True)
        r=tok(ex['rejected_text'],truncation=True,max_length=max_len,add_special_tokens=True)
        return {'chosen_input_ids':c['input_ids'],'chosen_attention_mask':c['attention_mask'],
                'rejected_input_ids':r['input_ids'],'rejected_attention_mask':r['attention_mask']}
    ds=ds.map(enc,remove_columns=ds['train'].column_names,desc='Tokenizing preference pairs')
    q=None
    if m.get('use_4bit',True):
        q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_use_double_quant=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16 if m.get('bf16',True) else torch.float16)
    local_rank=int(os.environ.get('LOCAL_RANK','0'))
    kwargs={'num_labels':1,'torch_dtype':torch.bfloat16 if m.get('bf16',True) else torch.float16,'quantization_config':q}
    if q is not None: kwargs['device_map']={'':local_rank}
    model=AutoModelForSequenceClassification.from_pretrained(base,local_files_only=True,**kwargs)
    model.config.pad_token_id=tok.pad_token_id; model.config.use_cache=False
    if q is not None: model=prepare_model_for_kbit_training(model)
    adapter=os.environ.get('INITIAL_ADAPTER','').strip()
    if adapter:
        # Recovery path: an epoch-only adapter has weights/tokenizer but no Trainer
        # state. Load it as the initialization and train the remaining epoch(s).
        peft=PeftModel.from_pretrained(model, adapter, is_trainable=True)
        print(f'INITIAL_ADAPTER={adapter}', flush=True)
    else:
        peft=get_peft_model(model,LoraConfig(task_type=TaskType.SEQ_CLS,r=int(l.get('r',32)),lora_alpha=int(l.get('alpha',64)),lora_dropout=float(l.get('dropout',.05)),target_modules=l.get('target_modules',['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']),bias='none'))
    peft.print_trainable_parameters()
    args=TrainingArguments(output_dir=t['rolling_dir'],num_train_epochs=float(os.environ.get('EPOCHS_OVERRIDE', t.get('num_train_epochs',2))),max_steps=int(os.environ.get('MAX_STEPS','-1')),learning_rate=float(t.get('learning_rate',1e-5)),weight_decay=float(t.get('weight_decay',.01)),warmup_ratio=float(t.get('warmup_ratio',.03)),per_device_train_batch_size=int(t.get('per_device_train_batch_size',1)),per_device_eval_batch_size=int(t.get('per_device_eval_batch_size',1)),gradient_accumulation_steps=int(t.get('gradient_accumulation_steps',16)),logging_steps=int(t.get('logging_steps',10)),eval_strategy=t.get('eval_strategy','steps'),eval_steps=int(t.get('eval_steps',100)),save_strategy=t.get('save_strategy','steps'),save_steps=int(t.get('save_steps',100)),save_total_limit=int(t.get('save_total_limit',3)),bf16=bool(m.get('bf16',True)),fp16=bool(m.get('fp16',False)),gradient_checkpointing=bool(t.get('gradient_checkpointing',True)),gradient_checkpointing_kwargs={'use_reentrant': False} if bool(t.get('gradient_checkpointing',True)) else None,report_to=t.get('report_to','none'),run_name=t.get('run_name'),ddp_find_unused_parameters=False,remove_unused_columns=False,seed=int(t.get('seed',42)))
    trainer=PairwiseTrainer(model=peft,args=args,train_dataset=ds['train'],eval_dataset=ds['validation'],data_collator=PairCollator(tok))
    trainer.add_callback(EpochSaveCallback(t['checkpoint_dir'],tok))
    return trainer,tok

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('config'); args=ap.parse_args(); cfg=load_cfg(args.config)
    set_seed(int(cfg['train'].get('seed',42)))
    for k in ('rolling_dir','checkpoint_dir','final_dir'): Path(cfg['train'][k]).mkdir(parents=True,exist_ok=True)
    trainer,tok=build(cfg)
    last=None
    if cfg['train'].get('resume_if_possible',True):
        from transformers.trainer_utils import get_last_checkpoint
        last=get_last_checkpoint(cfg['train']['rolling_dir'])
    print(f'RANK={os.environ.get("RANK","0")} WORLD_SIZE={os.environ.get("WORLD_SIZE","1")} resume={last}',flush=True)
    trainer.train(resume_from_checkpoint=last)
    if trainer.is_world_process_zero():
        trainer.save_model(cfg['train']['final_dir']); tok.save_pretrained(cfg['train']['final_dir'])
        print('FINAL_SAVED',cfg['train']['final_dir'],flush=True)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
if __name__=='__main__': main()
