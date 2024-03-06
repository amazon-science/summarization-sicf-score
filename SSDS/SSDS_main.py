# copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import nltk  
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


import copy


##### start my revision ######
from transformers import EarlyStoppingCallback, IntervalStrategy

from ssds_util import set_config, tr_val_te_func, metric_util, sample_weighted_trainer
from data_process import preprocess_dataset
from ssds_util.decode_sampling_func import decode_sampling_helper
import torch, gc
import random

from SiCF import sicf_score, unc_metrics, draw_sicf


os.environ["CUDA_VISIBLE_DEVICES"]='0'

check_min_version("4.9.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")


try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    round_name: str = field(
        default="labeled",
        metadata={"help": "which round is in [labeled, unlabeled]"},
    )
    do_gen_pseudo_label: bool = field(
        default=False,
        metadata={"help": "whehter generate pseudo label for the unlabled data"},
    )
    search_sample_mode: Optional[str] = field(
        default=None,
        metadata={"help": "[None, beam_sample, top_kp_sample, diverse_beam]"},
    )
    search_sampling_num: Optional[int] = field(
        default=3,
        metadata={"help": "the sampled size of predicted sequences for each input text"},
    )
    diverse_beam_groups: Optional[int] = field(
        default=3,
        metadata={"help": "the sampled size of predicted sequences for each input text"},
    )
    search_sampling_temp: Optional[float]= field(
        default=0.9,
        metadata={"help": "the temparature size of predicted sequences for each input text"},
    )
    nlg_metric: Optional[str] = field(
        default=None,
        metadata={"help": "way to calculate nlg results' uncertainty: [SiCF(inner semantic invarance, extra converage, inner faithfulness, extra faithfulness, inner repetivenss), oracle]"},
    )
    search_sampling_text_save_name: Optional[str] = field(
        default='search_sampling_pseudo_text.json',
        metadata={"help": "the saved name of texts from search_sampling ids"},
    )
    use_parallel_sampling: Optional[bool] = field(
        default=False,
        metadata={"help": "whehter use the parallel to sampling"},
    )
    parallel_sampling_id : Optional[int] = field(
        default=None,
        metadata={"help": "the id to paralle, the sampling file is named as [search_sampling_text_save_name[:-5]] + str([parallel_sampling_id] + .json)"},
    )

    search_sampling_filter_mode: Optional[str] = field(
        default=None,
        metadata={"help": "the mode to filter out the data [select, weigh, select_weigh ]"},
    )
    search_sampling_filter_thrs: Optional[float] = field(
        default=None,
        metadata={"help": "the threshold to filter out the data, value range may change according to method"},
    )
    length_penalty: Optional[float] = field(
        default=2.0,
        metadata={"help": "the length penality in used in val/predict/gen_pseudo."},
    )


    # start parameter related to measure pseudo labels
    do_measure_pseudo_label: Optional[bool] = field(
        default=False,
        metadata={"help": "whehter measure the quality of pseudo labels"},
    )

    sein_model_name: Optional[str] = field(
        default='roberta-base',
        metadata={"help": "whehter measure the quality of pseudo labels"},
    )

    sampling_case: Optional[str] = field(
        default='1p5t',
        metadata={"help": "case of percent & temparature"},
    )
    sampling_summary_root: Optional[str] = field(
        default='./data/sampled_data/',
        metadata={"help": "the root of sampling summary"},
    )
    dataname: Optional[str] = field(
        default='newSAMSUM2',
        metadata={"help": "the data name"},
    )

    use_small: Optional[bool] = field(
        default=False,
        metadata={"help": "whehter use the small data to train"},
    )
    use_weight_search: Optional[bool] = field(
        default=False,
        metadata={"help": "whehter use the weight to search"},
    )
    sicf_mode: Optional[str] = field(
        default='sein_cov_fai',
        metadata={"help": "set whether use sein (semantic invariance), cov (coverage), and fai (faithfulness)"},
    )
    use_sicf_rank: Optional[bool] = field(
        default=False,
        metadata={"help": " True: use location rank; False: use score "},
    )

    # sein related
    cal_sein_emb: Optional[bool] = field(
        default=True,
        metadata={"help": "whehter calculate the sein_emb if sein is open"},
    )

    # coverage related
    cal_cov_emb: Optional[bool] = field(
        default=True,
        metadata={"help": "whehter calculate the cov_emb if cov is open"},
    )
    use_cov_count: Optional[bool] = field(
        default=True,
        metadata={"help": "whehter use the counts of tokens to weigh the cov"},
    )
    cov_batch_size: Optional[int] = field(
        default=32,
        metadata={"help": "the batch size used in the cov calculation"},
    )


    # faithfulnes related
    cal_fai_emb: Optional[bool] = field(
        default=True,
        metadata={"help": "whehter calculate the fai_emb if fai is open"},
    )
    use_fai_count: Optional[bool] = field(
        default=True,
        metadata={"help": "whehter use the counts of sentences to weigh the fai"},
    )
    fai_batch_size: Optional[int] = field(
        default=32,
        metadata={"help": "the batch size used in the fai calculation"},
    )
    fai_cal_type: Optional[str] = field(
        default='nli',
        metadata={"help": "the way to cal fai scores [nli: using nli & ]"},
    )


    unc_type: Optional[str] = field(
        default='bnn_dropout',
        metadata={"help": "Set what kind of unc type to be used. [bnn_dropout, bnn_mean, bnn_var, bnn_mean_dp, bnn_mean_var]"},
    )
    bnn_type: Optional[int] = field(
        default=0,
        metadata={"help": "Set what kind of unc type to be used. [0: predictive, 1: aleatoric , 2: epistemic ]"},
    )
    multi_cal_type: Optional[str] = field(
        default='sum',
        metadata={"help": "Set what kind of unc type to be used. [mean, sum] applied into the bnn_dropout/bnn_var"},
    )
    unc_save_prefix: Optional[str] = field(
        default='',
        metadata={"help": "Set what kind of unc type to be used. [bnn_dropout, bnn_mean]"},
    )

    # baseline ralted
    use_unc_baseline: Optional[str] = field(
        default=None,
        metadata={"help": "whether use the baseline [random, allzero, oracle]"},
    )
    allzero_rdseed: Optional[int] = field(
        default=0,
        metadata={"help": "the random seed to run allzero"},
    )

    sein_coeff: Optional[float] = field(
        default=0.33,
        metadata={"help": "the coeff of wei scores, smaller gives more weights"},
    )
    cov_coeff: Optional[float] = field(
        default=0.33,
        metadata={"help": "the coeff of wei scores, smaller gives more weights"},
    )
    fai_coeff: Optional[float] = field(
        default=0.33,
        metadata={"help": "the coeff of wei scores, smaller gives more weights"},
    )
    use_sicf_ssds: Optional[bool] = field(
        default=False,
        metadata={"help": "whether use the sicf in the trainnig process"},
    )
    select_ratio_on_sicf: Optional[int] = field(
        default=None,
        metadata={"help": "whether select a subset of the sicf weighted samples 0-100"},
    )



    cal_unc_oracle: Optional[bool] = field(
        default=False,
        metadata={"help": "whether calculate the oracle for the unc, usually used for the "},
    )

    ###### above is self-added##########

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )

    ###### below is self-added##########
    ini_train_file: Optional[str] = field(
        default=None, metadata={"help": "The init training data file (a jsonlines or csv file), which will be divided into two parts"}
    )
    div_train: bool = field(
        default=True,
        metadata={
            "help": "Whether to divide into labeled tr and unlabeled tr, labeled_round:true, unlabeled_round:false."
        },
    )

    merge_pseudo_unlabeled_train_file:Optional[str] = field(
        default=None, metadata={"help": "In a basic case: the labeled training data & processed training data w/o labels (a jsonlines or csv file). wo or with sicf scores in it"}
    )
    sicf_merge_pseudo_unlabeled_train_file:Optional[str] = field(
        default="sicf_merge_pseudo_unlabeled_train.json", metadata={"help": "In a The labeled training data & processed training data w/o labels (a jsonlines or csv file)."}
    )

    pseudo_unlabeled_train_file: Optional[str] = field(
        default=None, metadata={"help": "The merge/processed training data w/o labels (a jsonlines or csv file)."}
    )

    gt_unlabeled_train_file: Optional[str] = field(
        default=None, metadata={"help": "The processed training data w/o labels (a jsonlines or csv file)."}
    )
    labeled_train_ratio: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "the ratio to indicate the which parts of ini_train_file is used for labeled json file, the 1-* is used for unlabeled train json file"
        },
    )

    sicf_pse_unlabeled_train_file: Optional[str] = field(
        default="sicf_pse_unlabeled_train.json", metadata={"help": "The merge/processed training data w/o labels (a jsonlines or csv file)."}
    )


    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file w/ labels (a jsonlines or csv file) to the semisupervised learning, not the full given train file."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``, the default value is self.model.config.num_beams=6, when is None"
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length





def main(logger):

    # 0. parameter lists
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    #######below starts revision#################
    if model_args.round_name == 'labeled':

        data_args.pseudo_unlabeled_train_file = os.path.join(training_args.output_dir, data_args.pseudo_unlabeled_train_file)

        data_args.merge_pseudo_unlabeled_train_file = os.path.join(training_args.output_dir, data_args.merge_pseudo_unlabeled_train_file)
        if model_args.use_small:
            data_args.sicf_pse_unlabeled_train_file = os.path.join(training_args.output_dir, data_args.sicf_pse_unlabeled_train_file[:-5] + '_small.json')
            data_args.sicf_merge_pseudo_unlabeled_train_file = os.path.join(training_args.output_dir, data_args.sicf_merge_pseudo_unlabeled_train_file[:-5] + '_small.json')
        else:

            data_args.sicf_pse_unlabeled_train_file = os.path.join(training_args.output_dir, data_args.sicf_pse_unlabeled_train_file)

            data_args.sicf_merge_pseudo_unlabeled_train_file = os.path.join(training_args.output_dir, data_args.sicf_merge_pseudo_unlabeled_train_file)

    #######end starts revision#################

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    random.seed(training_args.seed)
    print("seed is :, ", training_args.seed)

    preprocess_dataset.divide_ini_train(data_args=data_args, model_args=model_args)


    if model_args.round_name == 'labeled':
        round_keyword = 'labrd_'
    else:
        round_keyword = 'unlrd_'
    generated_predictions_name_end = 'generated_predictions.txt'

    model_args.sampling_summary_path = preprocess_dataset.get_sampling_summary_path(
        sampling_case=model_args.sampling_case,
        sampling_summary_root=model_args.sampling_summary_root,
        dataname=model_args.dataname,
    )
    print(model_args.sampling_summary_path)
    assert 1==0

    #######end starts revision#################

    # 1. load dataset - 1.1 select datasets
    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        if data_args.gt_unlabeled_train_file is not None:
            data_files["unlabtr"] = data_args.gt_unlabeled_train_file
            extension = data_args.test_file.split(".")[-1]
        if  model_args.round_name == 'labeled' or model_args.use_sicf_ssds == False:
            raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
        elif model_args.round_name == 'unlabeled' and model_args.use_sicf_ssds == True:
            data_file_tr = data_files.pop("train", None)
            extension_tr = data_file_tr.split(".")[-1]
            raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
            raw_dataset_tr = load_dataset(extension_tr, data_files=data_file_tr, cache_dir=model_args.cache_dir)
            raw_datasets['train'] = raw_dataset_tr['train']
        else:
            raise ValueError(f"model_args.round_name={model_args.round_name} and model_args.use_sicf_ssds = {model_args.use_sicf_ssds} are incosistent!")




    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    config = set_config.expand_config(config=config, data_args=data_args) # L349-353


    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""


    column_names, dataset_columns, text_column, summary_column, max_target_length, padding = preprocess_dataset.get_dataset_stat(
        training_args=training_args,
        data_args=data_args,
        raw_datasets=raw_datasets,
        model=model
    )   

    preprocess_obj = preprocess_dataset.Preprocess_data(
        text_column=text_column,
        summary_column = summary_column,
        prefix=prefix,
        tokenizer=tokenizer,
        data_args=data_args,
        padding=padding,
        max_target_length = max_target_length
    )  

    # safety judge
    if model_args.use_sicf_ssds == True:
        assert model_args.round_name == 'unlabeled'


    train_dataset, eval_dataset, predict_dataset, unlabtr_dataset = preprocess_dataset.get_tr_val_te_set(
        training_args=training_args,
        data_args=data_args,
        model_args=model_args,
        raw_datasets=raw_datasets,
        column_names=column_names,
        tr_preprocess_function=preprocess_obj.preprocess_weighted_sample_function if (model_args.round_name == 'unlabeled' and model_args.use_sicf_ssds==True) else preprocess_obj.preprocess_function,
        preprocess_function=preprocess_obj.preprocess_function, # used for val & test & unlabeled
    )    


    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    metric = load_metric("rouge")


    summary_metric = metric_util.Metrics(
        tokenizer=tokenizer,
        data_args=data_args,
        metric=metric
    )


    if model_args.round_name == 'labeled' or model_args.use_sicf_ssds == False:

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=summary_metric.compute_metrics_wBERTscore if training_args.predict_with_generate else None,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
        )  
    elif model_args.round_name == 'unlabeled' and model_args.use_sicf_ssds == True:
        trainer = sample_weighted_trainer.SampleWeightedSeq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=summary_metric.compute_metrics_wBERTscore if training_args.predict_with_generate else None,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
        )
    else:
        raise ValueError(f"model_args.round_name = {model_args.round_name} and model_args.use_sicf_ssds={model_args.use_sicf_ssds} is inconsistent")

    # 2. train a SSDS model 
    if training_args.do_train:
        trainer = tr_val_te_func.train(
            training_args=training_args,
            data_args=data_args,
            last_checkpoint=last_checkpoint,
            trainer=trainer,
            train_dataset=train_dataset
        ) 



    # 3. eval results on val 
    results = {} 
    if training_args.do_eval:
        trainer, logger = tr_val_te_func.val(
            data_args=data_args,
            model_args=model_args,
            trainer=trainer,
            eval_dataset=eval_dataset,
            logger=logger
        ) 


    # 4. evaluatiion 
    if training_args.do_predict:
        pred_key_prefix = round_keyword + 'stdtest'
        pred_test_gen_pred_file_name = pred_key_prefix + '_' + generated_predictions_name_end
        trainer, logger = tr_val_te_func.test(
            training_args=training_args,
            data_args=data_args,
            model_args=model_args,
            trainer=trainer,
            predict_dataset=predict_dataset,
            tokenizer=tokenizer,
            logger=logger,
            metric_key_prefix=pred_key_prefix,
            test_gen_pred_file_name=pred_test_gen_pred_file_name
        )  

    # 5. use func_test to predict "trainer.predic" summarizations for unlabled data.
    if model_args.do_gen_pseudo_label:
        assert model_args.round_name == 'labeled'
        pred_key_prefix = round_keyword + 'unlabtr'
        pred_test_gen_pred_file_name = pred_key_prefix + '_' + generated_predictions_name_end
        dec_samp_helper = decode_sampling_helper(
            preprocess_obj=preprocess_obj,
            use_parallel_sampling=model_args.use_parallel_sampling,
            parallel_sampling_id=model_args.parallel_sampling_id
        )


        trainer, logger, filtered_sampled_id = tr_val_te_func.gen_psudo(
            training_args=training_args,
            data_args=data_args,
            model_args=model_args,
            trainer=trainer,
            predict_dataset=unlabtr_dataset,
            tokenizer=tokenizer,
            logger=logger,
            metric_key_prefix=pred_key_prefix,
            test_gen_pred_file_name=pred_test_gen_pred_file_name,
            dec_samp_helper=dec_samp_helper
        )

        # set back the random seed after the random sampling
        set_seed(training_args.seed)
        random.seed(training_args.seed)


        # merge pred_test_gen_pred_file_name into the data_args.unlabeled_train_file with summary_column
        preprocess_dataset.merge_gene_as_unlabtr_labels(data_args=data_args, training_args=training_args, pred_test_gen_pred_file_name=pred_test_gen_pred_file_name, filtered_sampled_id=filtered_sampled_id)

    gc.collect()
    torch.cuda.empty_cache()



    if model_args.do_measure_pseudo_label == True:
        assert model_args.round_name == 'labeled'
        pred_key_prefix = round_keyword + 'unlabtr'
        pred_test_gen_pred_file_name = pred_key_prefix + '_' + generated_predictions_name_end

        sicf_tool = sicf_score.SiCF_Tool(
            save_path=training_args.output_dir,
            use_cov_count=model_args.use_cov_count,
            use_fai_count=model_args.use_fai_count,
            use_small=model_args.use_small,
            prefix=model_args.sampling_case+'_'+model_args.dataname,
            fai_cal_type=model_args.fai_cal_type,
            sein_dis='euclidean',
            cov_dis='euclidean', 
            fai_dis='euclidean', 
        )

        # extract txt
        if model_args.sicf_mode is not None:
            dialogue_list = preprocess_dataset.read_json(
                json_path=data_args.gt_unlabeled_train_file,
                use_small=model_args.use_small,
                keyword='src'
            )
            dialogue_list = sicf_tool.preprocess_text(dialogue_list)

        # extract summary
        sampling_summary_list = preprocess_dataset.read_sampling_json(
            json_path=model_args.sampling_summary_path,
            use_small=model_args.use_small,
            keyword='pred'
        )

        sein_wei = 0
        sein_save_path = os.path.join(training_args.output_dir, 'sein_stat.npz')
        if 'sein' in model_args.sicf_mode:
            # cal semantic inv
            sem_inv_mic, sem_inv_pool, sem_dist_arr, sem_disrank_list = sicf_tool.txt_semantic_invariance(txt_list=sampling_summary_list,
                                                                                                          cal_sein_emb=model_args.cal_sein_emb,
                                                                                                          model_name=model_args.sein_model_name)
            # sem_inv_mic: b * d arraies, where each d-dim array shows semantic invarance for a sample in a micro-level
            # sem_inv_pool: b-length list of scalars, each scalar shows semantic invarance for a sample
            # sem_dist_arr: b * m arraies, where each row shows the distance from m samples to the
            # sem_disrank_list: b-length list of scalars, each scalar shows semantic invarance for a sample


            # save score

            np.savez(sein_save_path,
                     sem_inv_mic=sem_inv_mic,
                     sem_inv_pool=sem_inv_pool,
                     sem_dist_arr=sem_dist_arr,
                     sem_disrank_list=sem_disrank_list
                     )


            sein_wei = sem_dist_arr[list(range(sem_dist_arr.shape[0])), sem_disrank_list]
            sein_wei = np.array(sein_wei)

            if model_args.use_small:
                print(f"sein_wei is {sein_wei}")

            print('finished semantic invariance scoring!')
        else: 
            if not os.path.isfile(sein_save_path): raise ValueError(f"The {sein_save_path} is not exist.")
            sein_dict = np.load(sein_save_path)
            sem_dist_arr = sein_dict['sem_dist_arr']
            sem_disrank_list = sein_dict['sem_disrank_list']

        if model_args.use_unc_baseline == 'random':
            sem_disrank_list = preprocess_dataset.generate_random_sampling_index(sampling_summary_list, training_args.seed)
            print('finished semantic invariance scoring!')


        if model_args.use_small:
            print(f"sem_disrank_list is {sem_disrank_list}")





        ## coverage
        cov_wei = 0
        if 'cov' in model_args.sicf_mode:

            # cal a list (#round) of cov score list (#sample)
            all_cov_score_list = sicf_tool.txt_coverage(src_list=sampling_summary_list,
                                   ref=dialogue_list,
                                   cal_cov_emb=model_args.cal_cov_emb,
                                   batch_size=model_args.cov_batch_size,
                                   )

            # 
            if sicf_tool.cov_dis in ['euclidean']:
                use_reciprocal=True
            elif sicf_tool.cov_dis in ['cosine']:
                use_reciprocal=True # due to 1-

            # calculate the uncertainty scores
            if model_args.unc_type == 'bnn_dropout':
                
                unc_cov_score_list = unc_metrics.multilabel_bnn_dp_cal_unc_score_list(all_cov_score_list, use_reciprocal=use_reciprocal, mode=model_args.multi_cal_type)

                cov_wei = unc_metrics.multilabel_read_uncer_score_list(unc_cov_score_list, index_enc=model_args.bnn_type)
            elif model_args.unc_type == 'bnn_var':
                cov_wei = unc_metrics.multilabel_bnn_var_cal(all_cov_score_list, mode=model_args.multi_cal_type)
            elif model_args.unc_type == 'bnn_mean':
                unc_cov_score_list, cov_wei = unc_metrics.bnn_mean_cal_unc_score_list(all_cov_score_list)
            elif model_args.unc_type == 'bnn_mean_dp':
                unc_cov_score_list_mean, cov_wei_mean = unc_metrics.bnn_mean_cal_unc_score_list(all_cov_score_list)

                unc_cov_score_list_dp = unc_metrics.multilabel_bnn_dp_cal_unc_score_list(all_cov_score_list, use_reciprocal=use_reciprocal, mode=model_args.multi_cal_type)
                cov_wei_dp = unc_metrics.multilabel_read_uncer_score_list(unc_cov_score_list_dp, index_enc=model_args.bnn_type)
                cov_wei = list(np.array(cov_wei_mean) * np.array(cov_wei_dp))
            elif model_args.unc_type == 'bnn_mean_var':
                unc_cov_score_list_mean, cov_wei_mean = unc_metrics.bnn_mean_cal_unc_score_list(all_cov_score_list)

                cov_wei_var = unc_metrics.multilabel_bnn_var_cal(all_cov_score_list, mode=model_args.multi_cal_type)
                cov_wei = list(np.array(cov_wei_mean) * np.array(cov_wei_var))


            cov_wei = np.array(cov_wei)
            if model_args.use_small:
                print(f"cov_wei is {cov_wei}")
            print('finished coverage scoring!')

        ## (to do) faithfulness
        fai_wei = 0
        if 'fai' in model_args.sicf_mode:

            
            all_fai_score_list = sicf_tool.txt_faithfulness(src_list=sampling_summary_list,
                                   ref=dialogue_list,
                                   cal_fai_emb=model_args.cal_fai_emb,
                                   batch_size=model_args.fai_batch_size,
                                   )

            if sicf_tool.fai_dis in ['euclidean']:
                use_reciprocal=True
            elif sicf_tool.fai_dis in ['cosine']:
                use_reciprocal=True # due to 1-

            # calculate the uncertainty scores
            if model_args.unc_type=='bnn_dropout':

                unc_fai_score_list = unc_metrics.multilabel_bnn_dp_cal_unc_score_list(all_fai_score_list, use_reciprocal=use_reciprocal, mode=model_args.multi_cal_type)
                # read the uncertainty scores
                fai_wei = unc_metrics.multilabel_read_uncer_score_list(unc_fai_score_list, index_enc=model_args.bnn_type)

            elif model_args.unc_type == 'bnn_var':
                fai_wei = unc_metrics.multilabel_bnn_var_cal(all_fai_score_list, mode=model_args.multi_cal_type)
            elif model_args.unc_type == 'bnn_mean':
                unc_fai_score_list, fai_wei = unc_metrics.bnn_mean_cal_unc_score_list(all_fai_score_list)
            elif model_args.unc_type == 'bnn_mean_dp':
                unc_fai_score_list_mean, fai_wei_mean = unc_metrics.bnn_mean_cal_unc_score_list(all_fai_score_list)
                unc_fai_score_list_dp = unc_metrics.multilabel_bnn_dp_cal_unc_score_list(all_fai_score_list, use_reciprocal=use_reciprocal, mode=model_args.multi_cal_type)
                fai_wei_dp = unc_metrics.multilabel_read_uncer_score_list(unc_fai_score_list_dp, index_enc=model_args.bnn_type)
                fai_wei = list(np.array(fai_wei_mean) * np.array(fai_wei_dp))
            elif model_args.unc_type == 'bnn_mean_var':
                unc_fai_score_list_mean, fai_wei_mean = unc_metrics.bnn_mean_cal_unc_score_list(all_fai_score_list)

                fai_wei_var = unc_metrics.multilabel_bnn_var_cal(all_fai_score_list, mode=model_args.multi_cal_type)
                fai_wei = list(np.array(fai_wei_mean) * np.array(fai_wei_var))


            fai_wei = np.array(fai_wei)
            if model_args.use_small:
                print(f"fai_wei is {fai_wei}")
            print('finished faithfulness scoring!')


        ## organize the gene by the sem_disrank_list
        preprocess_dataset.orga_gene_as_unlabtr_labels(data_args=data_args,
                                                       training_args=training_args,
                                                       model_args=model_args,
                                                       # pred_test_gen_pred_file_name=pred_test_gen_pred_file_name,
                                                       sampling_summary_list=sampling_summary_list,
                                                       sem_disrank_list=sem_disrank_list)

        ### evaluate the sicf score performance in uncertainty estimation
        if model_args.use_sicf_rank == True:
            sein_rank = sicf_tool.wei2rank(sein_wei)
            cov_rank = sicf_tool.wei2rank(cov_wei)
            fai_rank = sicf_tool.wei2rank(fai_wei)
            total_rank = model_args.sein_coeff * sicf_tool.get_total_rank(sein_rank) + model_args.cov_coeff * sicf_tool.get_total_rank(cov_rank) + model_args.fai_coeff * sicf_tool.get_total_rank(fai_rank)
            SiCF_score = (model_args.sein_coeff * sein_rank + model_args.cov_coeff * cov_rank + model_args.fai_coeff * fai_rank) / (total_rank + 1e-8)

        else:
            normalize_sein = sicf_tool.normalize_wei(sein_wei)
            normalize_cov = sicf_tool.normalize_wei(cov_wei)
            normalize_fai = sicf_tool.normalize_wei(fai_wei)
            total_coef = sicf_tool.get_total_coef(sein_wei, model_args.sein_coeff) + sicf_tool.get_total_coef(cov_wei, model_args.cov_coeff) + sicf_tool.get_total_coef(fai_wei, model_args.fai_coeff)
            SiCF_score = (model_args.sein_coeff * normalize_sein + model_args.cov_coeff * normalize_cov + model_args.fai_coeff * normalize_fai) / (total_coef + 1e-8)


        gt_summ, pse_summ = preprocess_dataset.sicf_eval_load_data(data_args=data_args,
                                                                   model_args=model_args
                                                                   )

        gt_summ = sicf_tool.preprocess_text(gt_summ)
        pse_summ = sicf_tool.preprocess_text(pse_summ)

        if model_args.use_weight_search:
            assert model_args.sicf_mode == "sein_cov_fai"
            search_list = [0, 0.25, 0.5, 0.75, 1.0]
            if model_args.use_small:
                search_list = [0, 0.2]
            search_jsq = 0
            search_res_dict = {}
            for sein_coeff in search_list:
                for cov_coeff in search_list:
                    for fai_coeff in search_list:
                        if model_args.use_sicf_rank == True:
                            SiCF_searched_score = (sein_coeff * sein_rank + cov_coeff * cov_rank + fai_coeff * fai_rank) / (sein_coeff + cov_coeff + fai_coeff + 1e-6)
                        else:
                            SiCF_searched_score = (sein_coeff * normalize_sein + cov_coeff * normalize_cov + fai_coeff * normalize_fai) / (sein_coeff + cov_coeff + fai_coeff + 1e-6)
                        SiCF_searched_score = list(SiCF_searched_score)
                        searched_metric_score, _ = unc_metrics.force_true_eval(
                            gt_seq_list=gt_summ,
                            pse_seq_list=pse_summ,
                            unc_socre=SiCF_searched_score,
                            metric_name='Rouge', # do not conduct BERTScore in this search
                            res_save_path=training_args.output_dir,
                            res_save_name=model_args.unc_save_prefix + '_' + 'sicf_res.json',  # saved json file name
                            xlsx_save_name=model_args.unc_save_prefix + '_' + 'force_eval_res.xlsx',
                            save_res=False,
                        )
                        cur_mean_rouge1 = np.array([ele['rouge1'] for ele in searched_metric_score]).mean()
                        search_res_dict[f"{sein_coeff} {cov_coeff} {fai_coeff}"] = cur_mean_rouge1
                        print(f'search_jsq={search_jsq}')
                        search_jsq += 1


            # analyze & print the best results
            print(search_res_dict) # (to do) save the dict
            search_res_dict_save_name = os.path.join(training_args.output_dir, model_args.unc_save_prefix+'_weight_search_dict.npy')
            np.save(search_res_dict_save_name, search_res_dict)
            search_res_dict2 = np.load(search_res_dict_save_name, allow_pickle=True)
            print(search_res_dict2)
            print('weight search finished!')
            # assert 1==0

        SiCF_score = list(SiCF_score)


        if model_args.use_small:
            print(SiCF_score)


        if model_args.use_unc_baseline != None:
            if model_args.use_unc_baseline == 'random':
                SiCF_score_len = len(SiCF_score)
                SiCF_score = [random.random() for _ in range(SiCF_score_len)]
            elif model_args.use_unc_baseline == 'allzero':
                SiCF_score_len = len(SiCF_score)
                SiCF_score = [random.random() for _ in range(SiCF_score_len)]
                set_seed(training_args.seed)
                random.seed(training_args.seed)
            elif model_args.use_unc_baseline == 'oracle':
                Oracle_score_based_bertscore = unc_metrics.cal_unc_oracle(
                    gt_seq_list=gt_summ,
                    pse_seq_list=pse_summ,
                )
                Oracle_score_based_bertscore = np.array(Oracle_score_based_bertscore)
                normalize_oracle = sicf_tool.normalize_wei(Oracle_score_based_bertscore)
                SiCF_score = normalize_oracle
            else:
                raise ValueError(f"model_args.use_unc_baseline={model_args.use_unc_baseline} is worngly set")

        if model_args.use_small:

            sicf_tool.check_txt_coverage_quality(
                src_list=sampling_summary_list,
                ref=dialogue_list,
                gt=gt_summ,
                batch_size=1
            )

            print('below is gt_summ')
            for ele in gt_summ:
                print(ele)

            print('below is pse_summ')
            for ele in pse_summ:
                print(ele)


        print(SiCF_score)

        draw_sicf.draw_fig(SiCF_score, save_fig_name='./img/' + model_args.unc_save_prefix + '_sample_weight.pdf')


        preprocess_dataset.add_save_sample_weight(data_args=data_args, training_args=training_args, model_args=model_args, sample_weight=SiCF_score)


        # simulate force-true evaluation
        unc_rouge_bert_score, unc_res_save_file = unc_metrics.force_true_eval(
            gt_seq_list=gt_summ,
            pse_seq_list=pse_summ,
            unc_socre=SiCF_score, 
            metric_name='Rouge_BERTScore',
            res_save_path=training_args.output_dir,
            res_save_name=model_args.unc_save_prefix+'_'+ 'sicf_res.json', 
            xlsx_save_name=model_args.unc_save_prefix+ '_' + 'force_eval_res.xlsx', 
        )

        if model_args.select_ratio_on_sicf is not None:
            preprocess_dataset.select_subset_on_sicf(data_args=data_args, training_args=training_args, model_args=model_args, sample_weight=SiCF_score)
            pass


        if model_args.cal_unc_oracle:
            Oracle_score_based_bertscore = unc_metrics.cal_unc_oracle(
                gt_seq_list=gt_summ,
                pse_seq_list=pse_summ,
            )
            unc_rouge_bert_score, unc_res_save_file = unc_metrics.force_true_eval(
                gt_seq_list=gt_summ,
                pse_seq_list=pse_summ,
                unc_socre=Oracle_score_based_bertscore,
                metric_name='Rouge_BERTScore',
                res_save_path=training_args.output_dir,
                res_save_name=model_args.unc_save_prefix + '_' + 'oracle_sicf_res.json',  # saved json file name
                xlsx_save_name=model_args.unc_save_prefix + '_' + 'oracle_force_eval_res.xlsx',  # saved xlsx file name
            )

    return results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    main(logger)



