import json
import random
import os
from shutil import copyfile

use_small_sample_num = 16
use_small_round_size = 3

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}

def get_dataset_stat(training_args, data_args, raw_datasets, model):
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )
    return column_names, dataset_columns, text_column, summary_column, max_target_length, padding


class Preprocess_data(object):
    def __init__(self, text_column, summary_column, prefix, tokenizer, data_args, padding, max_target_length):
        self.text_column = text_column
        self.summary_column = summary_column
        self.prefix = prefix
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.padding = padding
        self.max_target_length = max_target_length


    def preprocess_function(self, examples):
        inputs = examples[self.text_column]
        targets = examples[self.summary_column]
        inputs = [self.prefix + inp for inp in inputs]
        model_inputs = self.tokenize_input(inputs)


        labels = self.tokenize_label(targets)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs




    def preprocess_weighted_sample_function(self, examples):
        inputs = examples[self.text_column]
        targets = examples[self.summary_column]
        inputs = [self.prefix + inp for inp in inputs]
        model_inputs = self.tokenize_input(inputs)

        labels = self.tokenize_label(targets)

        sample_weights = examples['sample_weight']
        model_inputs['sample_weight'] = sample_weights

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    def tokenize_input(self, inputs):
        res = self.tokenizer(inputs, max_length=self.data_args.max_source_length, padding=self.padding, truncation=True)
        return res

    def tokenize_label(self, targets):
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.max_target_length, padding=self.padding, truncation=True)

        if self.padding == "max_length" and self.data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        return labels




def get_tr_val_te_set(training_args, data_args, model_args, raw_datasets, column_names, tr_preprocess_function, preprocess_function):
    train_dataset, eval_dataset, predict_dataset, unlabtr_dataset  = None, None, None, None
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]

        if model_args.use_sicf_ssds == True:
            assert 'sample_weight' in train_dataset.column_names

        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(

                tr_preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        if 'sample_weight' in column_names:
            column_names.pop(column_names.index('sample_weight'))

        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        if 'sample_weight' in column_names:
            column_names.pop(column_names.index('sample_weight'))

        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    if model_args.do_gen_pseudo_label:
        if 'sample_weight' in column_names: # should not happen
            column_names.pop(column_names.index('sample_weight'))

        max_target_length = data_args.val_max_target_length
        if "unlabtr" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        unlabtr_dataset = raw_datasets["unlabtr"]
        if data_args.max_predict_samples is not None:
            unlabtr_dataset = unlabtr_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="unlabeled dataset map pre-processing"):
            unlabtr_dataset = unlabtr_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on unlabeled dataset",
            )
    return train_dataset, eval_dataset, predict_dataset, unlabtr_dataset


#### below is self-added ######
def divide_ini_train(data_args, model_args):
    # input:
    # output:
    #       labeled_tr_file: generaed a file with labeled_tr data
    #       labeled_tr_file: generaed a file with unlabeled_tr data

    if model_args.round_name == 'labeled':
        if "SUM" in data_args.ini_train_file or "SUM" in data_args.train_file:
            assert data_args.div_train == False # because "proSAMSUM" is preprocessed to have the preset data split by "extract_data_split.py"
        elif data_args.ini_train_file != data_args.train_file:
            if os.path.isfile(data_args.train_file) == True:
                assert data_args.div_train == False 
            else:
                assert data_args.div_train == True 
    elif model_args.round_name == 'unlabeled':
        assert data_args.div_train == False
        assert data_args.pseudo_unlabeled_train_file is None
    else:
        raise ValueError('the model_args.round_name is wrongly set [labeled, unlabeled]')

    if data_args.dataset_name is not None:
        raise NotImplementedError('have not considered that data_args.dataset_name is not None')

    if data_args.div_train:
        labeled_datasets, unlabeled_datasets = None, None
        if data_args.ini_train_file == data_args.train_file:
            print('if data_args.ini_train_file == data_args.train_file, then the original data will be overwritting')
            raise ValueError('if want to use the ori real_data, should set data_args.div_train=False')
        ini_data_file = data_args.ini_train_file
        extension = data_args.train_file.split(".")[-1]

        if 'json' in extension:

            all_tr_list = json_to_list(file_name=ini_data_file)
            random.Random(0).shuffle(all_tr_list) # random shuffle

            all_tr_num = len(all_tr_list)
            labeled_tr_list = all_tr_list[0: int(all_tr_num*data_args.labeled_train_ratio)]
            unlabeled_tr_list = all_tr_list[int(all_tr_num*data_args.labeled_train_ratio):]

            labeled_tr_file, unlabeled_tr_file = data_args.train_file, data_args.gt_unlabeled_train_file
            list_to_json(p_list=labeled_tr_list, file_name=labeled_tr_file)
            list_to_json(p_list=unlabeled_tr_list, file_name=unlabeled_tr_file)

        else:
            raise NotImplementedError(f'the {extension} file has not been implemented')
    else:
        if not os.path.isfile(data_args.train_file):
            raise ValueError(f"data_args.div_train=False, then {data_args.train_file} should exist")




def add_save_sample_weight(data_args, training_args, model_args, sample_weight=None):
    # merge pred_test_gen_pred_file_name into the data_args.unlabeled_train_file with summary_column
    summary_column = data_args.summary_column
    gt_unlabeled_tr_file = data_args.gt_unlabeled_train_file

    
    sicf_pse_unlabeled_tr_file = data_args.sicf_pse_unlabeled_train_file # have the selected psueudo labels for each unlabeled data
    assert os.path.isfile(sicf_pse_unlabeled_tr_file)

    if sample_weight is None:
        return
    else:
        pse_unlabeled_tr_wi_sw_file = data_args.pseudo_unlabeled_train_file[:-5] + '_wi_sampleweight' + '_' + model_args.unc_save_prefix + '.json'
        sampleweight_column = 'sample_weight'
        sample_weight = list(sample_weight)



    all_tr_list = json_to_list(sicf_pse_unlabeled_tr_file)


    # assert sample_weight is not None
    if sample_weight is not None:
        # add weight as sicf score (smaller is better)
        print(f"len(all_tr_list)={len(all_tr_list)}")
        print(f"len(sample_weight)={len(sample_weight)}")
        for i in range(len(all_tr_list)):
            all_tr_list[i][sampleweight_column] = sample_weight[i] # sicf score


    # save the sample weight
    list_to_json(p_list=all_tr_list, file_name=pse_unlabeled_tr_wi_sw_file) # save unlabled data with psuedo summaries
    print(f'the {pse_unlabeled_tr_wi_sw_file} has add weights based on {sicf_pse_unlabeled_tr_file}.')

    # below is for the merge operation

    merge_pse_unlabeled_tr_wi_sw_file = data_args.merge_pseudo_unlabeled_train_file[:-5] + '_wi_sampleweight' + '_' + model_args.unc_save_prefix + '.json'
    labeled_tr_file = data_args.train_file
    labled_tr_list = json_to_list(labeled_tr_file) # load labeled data

    if sample_weight is not None:
        for i in range(len(labled_tr_list)):
            labled_tr_list[i][sampleweight_column] = 0.0  # sicf score

    merge_tr_list = []
    merge_tr_list.extend(labled_tr_list)
    merge_tr_list.extend(all_tr_list)
    # shuffle
    random.Random(0).shuffle(merge_tr_list)
    list_to_json(p_list=merge_tr_list, file_name=merge_pse_unlabeled_tr_wi_sw_file)
    if sample_weight is not None:
        print(f'sample weight is effective, the {merge_pse_unlabeled_tr_wi_sw_file} has merged {pse_unlabeled_tr_wi_sw_file}   and {labeled_tr_file}.')



def select_subset_on_sicf(data_args, training_args, model_args, sample_weight):
    # select a subset of the weight sicf based on the sicf rank and the given ratio
    assert model_args.select_ratio_on_sicf <= 100
    summary_column = data_args.summary_column
    gt_unlabeled_tr_file = data_args.gt_unlabeled_train_file


    sicf_pse_unlabeled_tr_file = data_args.sicf_pse_unlabeled_train_file # have the selected psueudo labels for each unlabeled data
    assert os.path.isfile(sicf_pse_unlabeled_tr_file)

    if sample_weight is None:
        print('need to add code to read sample wegiht from the wi_weight json file')
        return
    else:
        pse_unlabeled_tr_wi_sw_file = data_args.pseudo_unlabeled_train_file[:-5] + '_wi_sampleweight' + '_' + model_args.unc_save_prefix + f'_select_{model_args.select_ratio_on_sicf}' + '.json'
        sampleweight_column = 'sample_weight'
        sample_weight = list(sample_weight)

    sample_weight_sort_idx = sorted(range(len(sample_weight)), key=lambda k:sample_weight[k], reverse=False) # False: small to large
    selected_num = int(len(sample_weight) * model_args.select_ratio_on_sicf / 100)
    selected_ids = sample_weight_sort_idx[0: selected_num]


    all_tr_list = json_to_list(sicf_pse_unlabeled_tr_file)

    # add weight as sicf score (smaller is better)
    print(f"len(all_tr_list)={len(all_tr_list)}")
    print(f"len(sample_weight)={len(sample_weight)}")
    print(f"len(selected_ids)={len(selected_ids)}")
    select_all_tr_list = []
    for i in selected_ids:

        mid_res = all_tr_list[i]
        mid_res[sampleweight_column] = 0.0
        select_all_tr_list.append(mid_res)


    # save the sample weight
    list_to_json(p_list=select_all_tr_list, file_name=pse_unlabeled_tr_wi_sw_file) # save unlabled data with psuedo summaries
    print(f'the {pse_unlabeled_tr_wi_sw_file} has selected weights based on in ratio of {model_args.select_ratio_on_sicf}.')

    # below is for the merge operation

    merge_pse_unlabeled_tr_wi_sw_file = data_args.merge_pseudo_unlabeled_train_file[:-5] + '_wi_sampleweight' + '_' + model_args.unc_save_prefix + f'_select_{model_args.select_ratio_on_sicf}' + '.json'
    labeled_tr_file = data_args.train_file
    labled_tr_list = json_to_list(labeled_tr_file) # load labeled data

    if sample_weight is not None:
        for i in range(len(labled_tr_list)):
            labled_tr_list[i][sampleweight_column] = 0.0  # sicf score

    merge_tr_list = []
    merge_tr_list.extend(labled_tr_list)
    merge_tr_list.extend(select_all_tr_list)
    # shuffle
    random.Random(0).shuffle(merge_tr_list)
    list_to_json(p_list=merge_tr_list, file_name=merge_pse_unlabeled_tr_wi_sw_file)
    if sample_weight is not None:
        print(f'sample weight is effective, the {merge_pse_unlabeled_tr_wi_sw_file} has merged {pse_unlabeled_tr_wi_sw_file}   and {labeled_tr_file}.')







def merge_gene_as_unlabtr_labels(data_args, training_args, pred_test_gen_pred_file_name, filtered_sampled_id):
    # merge pred_test_gen_pred_file_name into the data_args.unlabeled_train_file with summary_column
    summary_column = data_args.summary_column
    gt_unlabeled_tr_file = data_args.gt_unlabeled_train_file
    pse_unlabeled_tr_file = data_args.pseudo_unlabeled_train_file
    test_gen_pred_file = os.path.join(training_args.output_dir, pred_test_gen_pred_file_name) # might need change in diverse generation cases

    if filtered_sampled_id == None: # have no filter, then normal
        copyfile(gt_unlabeled_tr_file, pse_unlabeled_tr_file)
        copyfile(gt_unlabeled_tr_file, pse_unlabeled_tr_file[:-5] + '_filter_ground_truth.json')
    else:
        filter_unlab_gt = []
        all_unlab_gt_list = json_to_list(gt_unlabeled_tr_file)
        for summmary_i in filtered_sampled_id:
            filter_unlab_gt.append(all_unlab_gt_list[summmary_i])
        list_to_json(p_list=filter_unlab_gt, file_name=pse_unlabeled_tr_file[:-5] + '_filter_ground_truth.json')

        # filter the filter_ground_truth
        copyfile(pse_unlabeled_tr_file[:-5] + '_filter_ground_truth.json', pse_unlabeled_tr_file)

    all_tr_list = json_to_list(pse_unlabeled_tr_file)
    with open(test_gen_pred_file, 'r') as f:
        gene_summaries = f.readlines()
    assert len(gene_summaries) == len(all_tr_list)
    for i in range(len(all_tr_list)):
        all_tr_list[i][summary_column] = ""
        all_tr_list[i][summary_column] = gene_summaries[i].strip('\n')
    list_to_json(p_list=all_tr_list, file_name=pse_unlabeled_tr_file) # save unlabled data with psuedo summaries
    print(f'the {pse_unlabeled_tr_file} has replaced   {summary_column}   column by {test_gen_pred_file}.')

    # below is for the merge operation
    merge_pse_unlabeled_tr_file = data_args.merge_pseudo_unlabeled_train_file
    labeled_tr_file = data_args.train_file
    labled_tr_list = json_to_list(labeled_tr_file) # load labeled data
    merge_tr_list = []
    merge_tr_list.extend(labled_tr_list)
    merge_tr_list.extend(all_tr_list)
    # shuffle
    random.Random(0).shuffle(merge_tr_list)
    list_to_json(p_list=merge_tr_list, file_name=merge_pse_unlabeled_tr_file)
    print(f'the {merge_pse_unlabeled_tr_file} has merged {pse_unlabeled_tr_file}   and {labeled_tr_file}.')




def merge_gene_as_unlabtr_labels_cp_before_add_filtered_sampled_id(data_args, training_args, pred_test_gen_pred_file_name):
    # merge pred_test_gen_pred_file_name into the data_args.unlabeled_train_file with summary_column
    summary_column = data_args.summary_column
    gt_unlabeled_tr_file = data_args.gt_unlabeled_train_file
    pse_unlabeled_tr_file = data_args.pseudo_unlabeled_train_file
    test_gen_pred_file = os.path.join(training_args.output_dir, pred_test_gen_pred_file_name) # might need change in diverse generation cases
    from shutil import copyfile
    try:
        copyfile(gt_unlabeled_tr_file, pse_unlabeled_tr_file)
    except IOError as e:
        print(f"Unable to copy {gt_unlabeled_tr_file} as {e}")
    all_tr_list = json_to_list(pse_unlabeled_tr_file)
    with open(test_gen_pred_file, 'r') as f:
        gene_summaries = f.readlines()
    assert len(gene_summaries) == len(all_tr_list)
    for i in range(len(all_tr_list)):
        all_tr_list[i][summary_column] = ""
        all_tr_list[i][summary_column] = gene_summaries[i].strip('\n')
    list_to_json(p_list=all_tr_list, file_name=pse_unlabeled_tr_file) # save unlabled data with psuedo summaries
    print(f'the {pse_unlabeled_tr_file} has replaced   {summary_column}   column by {test_gen_pred_file}.')

    # below is for the merge operation
    merge_pse_unlabeled_tr_file = data_args.merge_pseudo_unlabeled_train_file
    labeled_tr_file = data_args.train_file
    labled_tr_list = json_to_list(labeled_tr_file) # load labeled data
    merge_tr_list = []
    merge_tr_list.extend(labled_tr_list)
    merge_tr_list.extend(all_tr_list)
    # shuffle
    random.Random(0).shuffle(merge_tr_list)
    list_to_json(p_list=merge_tr_list, file_name=merge_pse_unlabeled_tr_file)
    print(f'the {merge_pse_unlabeled_tr_file} has merged {pse_unlabeled_tr_file}   and {labeled_tr_file}.')

def orga_gene_as_unlabtr_labels(data_args, training_args, model_args, sampling_summary_list, sem_disrank_list):
    summary_column = data_args.summary_column
    gt_unlabeled_tr_file = data_args.gt_unlabeled_train_file
    sicf_pse_unlabeled_tr_file = data_args.sicf_pse_unlabeled_train_file # a json file with the ranked ID
    from shutil import copyfile
    try:
        copyfile(gt_unlabeled_tr_file, sicf_pse_unlabeled_tr_file)
    except IOError as e:
        print(f"Unable to copy {gt_unlabeled_tr_file} as {e}")
    all_tr_list = json_to_list(sicf_pse_unlabeled_tr_file)

    if model_args.use_small:
        all_tr_list = all_tr_list[0:use_small_sample_num]

    gene_summaries = []
    sem_disrank_list = list(sem_disrank_list)
    print(f"sem_disrank_list has lenght of {len(sem_disrank_list)}")
    print(f"all_tr_list has lenght of {len(all_tr_list)}")
    assert len(sem_disrank_list) == len(all_tr_list)
    for k in range(len(sem_disrank_list)):
        cur_indx = sem_disrank_list[k]
        gene_summaries.append(sampling_summary_list[cur_indx][k])

    assert len(gene_summaries) == len(all_tr_list)
    for i in range(len(all_tr_list)):
        all_tr_list[i][summary_column] = ""
        all_tr_list[i][summary_column] = gene_summaries[i].strip('\n')
    list_to_json(p_list=all_tr_list,
                 file_name=sicf_pse_unlabeled_tr_file)  # save unlabled data with psuedo summaries
    print(f'the {sicf_pse_unlabeled_tr_file} has replaced   {summary_column}   column by {sampling_summary_list}.')

    # below is for the merge operation
    sicf_merge_pseudo_unlabeled_train_file = data_args.sicf_merge_pseudo_unlabeled_train_file
    labeled_tr_file = data_args.train_file
    labled_tr_list = json_to_list(labeled_tr_file)  # load labeled data
    merge_tr_list = []
    merge_tr_list.extend(labled_tr_list)
    merge_tr_list.extend(all_tr_list)
    # shuffle
    random.Random(0).shuffle(merge_tr_list)
    list_to_json(p_list=merge_tr_list, file_name=sicf_merge_pseudo_unlabeled_train_file)
    print(f'the {sicf_merge_pseudo_unlabeled_train_file} has merged {sicf_pse_unlabeled_tr_file}   and {labeled_tr_file}.')


def sicf_eval_load_data(data_args, model_args):
    gt_unlabeled_tr_file = data_args.gt_unlabeled_train_file
    gt_unlabeled_tr = json_to_list(gt_unlabeled_tr_file)

    sicf_pse_unlabeled_tr_file = data_args.sicf_pse_unlabeled_train_file
    sicf_pse_unlabeled_tr = json_to_list(sicf_pse_unlabeled_tr_file)

    summary_column = data_args.summary_column

    if model_args.use_small:
        gt_unlabeled_tr = gt_unlabeled_tr[0:use_small_sample_num]
        sicf_pse_unlabeled_tr = sicf_pse_unlabeled_tr[0:use_small_sample_num]

    gt_text = []
    sicf_pse_text = []
    for gt in gt_unlabeled_tr:
        gt_text.append(gt[summary_column])
    for sicf_pse in sicf_pse_unlabeled_tr:
        sicf_pse_text.append(sicf_pse[summary_column])

    return gt_text, sicf_pse_text

def json_to_list(file_name):
    all_tr_list = []
    try:
        # each row is a json object
        with open(file_name, 'r') as ini_f:
            ini_tr_files = ini_f.readlines()
            for tr_file in ini_tr_files:
                cur_tr = json.loads(tr_file)
                all_tr_list.append(cur_tr)
    except:
        # the whole file is a json object
        all_tr_list = json.loads(file_name)
    return all_tr_list


def list_to_json(p_list, file_name):
    if os.path.isfile(file_name):
        os.remove(file_name)
    with open(file_name, 'a') as f:
        for ele in p_list:
            json_str = json.dumps(ele)
            f.write(json_str)
            f.write('\n')
    print(f"{file_name} has been generated")


def read_json(json_path, use_small, keyword='src'):
    assert keyword != 'tgt' # 'tgt' is gt and should never be used
    sample_text = json_to_list(json_path)
    text_len = len(sample_text)
    res = []
    if use_small:
        text_len = use_small_sample_num
    for i in range(text_len):
        res.append(sample_text[i][keyword])
    return res



def read_sampling_json(json_path, use_small, keyword='pred'):
    with open(json_path, 'r') as f:
        sample_text = json.load(f)
    if use_small:
        sampling_time = use_small_round_size
        sample_size = use_small_sample_num
        res = []
        for i in range(sampling_time):
            res.append(sample_text[keyword][i][0:sample_size])
        return res
    else:
        return sample_text[keyword]


def generate_random_sampling_index(sampling_list, seed):
    round_num = len(sampling_list)
    sample_num = len(sampling_list[0])
    random.seed(seed)
    sample_list = list(range(round_num))
    res = []
    for i in range(sample_num):
        cur_index = random.choice(sample_list)
        res.append(cur_index)
    return res

def get_sampling_summary_path(sampling_case, sampling_summary_root, dataname):
    sampling_summary_path = os.path.join(sampling_summary_root, dataname)
    if sampling_case == '1p10t': # t = 1.0
        sampling_summary_file = 'search_sampling_pseudo_text_1p_t10.json'
    elif sampling_case == '5p10t':
        sampling_summary_file = 'search_sampling_pseudo_text_5p_t10.json'
    else:
        raise ValueError(f"sampling_case={sampling_case} is wrongly set!")
    return os.path.join(sampling_summary_path, sampling_summary_file)