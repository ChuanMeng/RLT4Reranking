import logging
import os
import sys
from contextlib import nullcontext

from tqdm import tqdm

import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import (HfArgumentParser,) # HfArgumentParser是Transformer框架中的命令行解析工具

from tevatron.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments

from data import HFRerankDataset, RerankerInferenceDataset, RerankerInferenceCollator
from modeling import RerankerModel

logger = logging.getLogger(__name__)


def main():

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError('Multi-GPU encoding is not supported.')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path, # tokenizer_name: meta-llama/Llama-2-7b-hf
        cache_dir = "/gpfs/work3/0/guse0654/cache/models/",
        token = "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",
    )


    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ???
    tokenizer.padding_side = 'right'

    model = RerankerModel.load(
        model_name_or_path=model_args.model_name_or_path, # castorini/rankllama-v1-7b-lora-passage
        torch_dtype=torch.float16 if training_args.fp16 else torch.float32, # fp16
        cache_dir = "/gpfs/work3/0/guse0654/cache/models/",
        token = "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",
    )

    rerank_dataset = HFRerankDataset(tokenizer=tokenizer, data_args=data_args, cache_dir=data_args.data_cache_dir or model_args.cache_dir)

    # q_max_len 32
    # p_max_len 164
    rerank_dataset = RerankerInferenceDataset(
        rerank_dataset.process(data_args.encode_num_shard, # 1
                               data_args.encode_shard_index), # 0
        tokenizer, max_q_len=data_args.q_max_len, max_p_len=data_args.p_max_len
    )

    rerank_loader = DataLoader(
        rerank_dataset,
        batch_size=training_args.per_device_eval_batch_size, # 64
        collate_fn=RerankerInferenceCollator(
            tokenizer,
            max_length=data_args.q_max_len+data_args.p_max_len,
            padding='max_length'
        ),
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers, # 0
    )

    """
    TevatronTrainingArguments(
    _n_gpu=1,
    adafactor=False,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-08,
    auto_find_batch_size=False,
    bf16=False,
    bf16_full_eval=False,
    data_seed=None,
    dataloader_drop_last=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=True,
    ddp_backend=None,
    ddp_broadcast_buffers=None,
    ddp_bucket_cap_mb=None,
    ddp_find_unused_parameters=None,
    ddp_timeout=1800,
    debug=[],
    deepspeed=None,
    disable_tqdm=False,
    dispatch_batches=None,
    do_encode=False,
    do_eval=False,
    do_predict=False,
    do_train=False,
    eval_accumulation_steps=None,
    eval_delay=0,
    eval_steps=None,
    evaluation_strategy=no,
    fp16=True,
    fp16_backend=auto,
    fp16_full_eval=False,
    fp16_opt_level=O1,
    fsdp=[],
    fsdp_config={'min_num_params': 0, 'xla': False, 'xla_fsdp_grad_ckpt': False},
    fsdp_min_num_params=0,
    fsdp_transformer_layer_cls_to_wrap=None,
    full_determinism=False,
    gc_p_chunk_size=32,
    gc_q_chunk_size=4,
    grad_cache=False,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    gradient_checkpointing_kwargs=None,
    greater_is_better=None,
    group_by_length=False,
    half_precision_backend=auto,
    hub_always_push=False,
    hub_model_id=None,
    hub_private_repo=False,
    hub_strategy=every_save,
    hub_token=<HUB_TOKEN>,
    ignore_data_skip=False,
    include_inputs_for_metrics=False,
    include_tokens_per_second=False,
    jit_mode_eval=False,
    label_names=None,
    label_smoothing_factor=0.0,
    learning_rate=5e-05,
    length_column_name=length,
    load_best_model_at_end=False,
    local_rank=0,
    log_level=passive,
    log_level_replica=warning,
    log_on_each_node=True,
    logging_dir=temp/runs/Nov15_17-59-30_gcn36.local.snellius.surf.nl,
    logging_first_step=False,
    logging_nan_inf_filter=True,
    logging_steps=500,
    logging_strategy=steps,
    lr_scheduler_kwargs={},
    lr_scheduler_type=linear,
    max_grad_norm=1.0,
    max_steps=-1,
    metric_for_best_model=None,
    mp_parameters=,
    neftune_noise_alpha=None,
    negatives_x_device=False,
    no_cuda=False,
    num_train_epochs=3.0,
    optim=adamw_torch,
    optim_args=None,
    output_dir=temp,
    overwrite_output_dir=False,
    past_index=-1,
    per_device_eval_batch_size=64,
    per_device_train_batch_size=8,
    prediction_loss_only=False,
    push_to_hub=False,
    push_to_hub_model_id=None,
    push_to_hub_organization=None,
    push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
    ray_scope=last,
    remove_unused_columns=True,
    report_to=['tensorboard'],
    resume_from_checkpoint=None,
    run_name=temp,
    save_on_each_node=False,
    save_safetensors=True,
    save_steps=500,
    save_strategy=steps,
    save_total_limit=None,
    seed=42,
    skip_memory_metrics=True,
    split_batches=False,
    tf32=None,
    torch_compile=False,
    torch_compile_backend=None,
    torch_compile_mode=None,
    torchdynamo=None,
    tpu_metrics_debug=False,
    tpu_num_cores=None,
    use_cpu=False,
    use_ipex=False,
    use_legacy_prediction_loop=False,
    use_mps_device=False,
    warmup_ratio=0.1,
    warmup_steps=0,
    weight_decay=0.0,
    )
    cuda:0
    """
    model = model.to(training_args.device)
    model.eval()

    all_results = {}


    for (batch_query_ids, batch_text_ids, batch) in tqdm(rerank_loader):
        with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            with torch.no_grad():
                for k, v in batch.items():
                    batch[k] = v.to(training_args.device)

                model_output = model(batch)

                scores = model_output.scores.cpu().detach().numpy()

                for i in range(len(scores)):
                    qid = batch_query_ids[i]
                    docid = batch_text_ids[i]
                    score = scores[i][0]

                    if qid not in all_results:
                        all_results[qid] = []

                    all_results[qid].append((docid, score))

    # {qid:[(docid, socre)...]}
    with open(data_args.encoded_save_path, 'w') as f:
        for qid in all_results:

            results = sorted(all_results[qid], key=lambda x: x[1], reverse=True)

            for idx, (docid, score) in enumerate(results):
                rank = idx+1
                f.write(f'{qid} Q0 {docid} {rank} {score} rankllama\n')

if __name__ == "__main__":
    main()
