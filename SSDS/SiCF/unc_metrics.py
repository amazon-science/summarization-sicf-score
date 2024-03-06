###################
# This metric file is revised on code of
# Towards more accurate uncertainty estimation in text classification, EMNLP 2020
# Uncertainty Aware Semi-Supervised Learning on Graph Data, NIPS 2020
#
###################

import json
import os

import numpy as np
from operator import itemgetter
import nltk  # Here to have a nice missing dependency error message early on
from bert_score import BERTScorer
from datasets import load_metric

from . import read_res_helper

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def force_true_eval(gt_seq_list, pse_seq_list, unc_socre, metric_name, res_save_path, res_save_name, xlsx_save_name, save_res=True):
    unc_socre = np.array(unc_socre)
    unc_max =  unc_socre.max()
    conf_score = unc_max - unc_socre
    conf_score = list(conf_score)
    if 'Rouge' not in metric_name and 'BERTScore' not in metric_name:
        raise ValueError(f"metric_name = {metric_name} is wrongly set!")

    if 'Rouge' in metric_name:
        rouge_metric = load_metric("rouge")
    if 'BERTScore' in metric_name:
        bertscorer = BERTScorer(lang="en", rescale_with_baseline=True)
        bert_metric = bertscorer.score

    pse_seq_list, gt_seq_list = postprocess_text(pse_seq_list, gt_seq_list)

    y_truth = gt_seq_list
    y_pred = pse_seq_list
    indices, L_sorted = zip(*sorted(enumerate(conf_score), key=itemgetter(1), reverse=True))

    metric_score = []


    idk_list = np.arange(0, 1.1, 0.1)
    for idk_ratio in idk_list:
        # print("=== idk_ratio: ", idk_ratio, " ===")
        test_num = int(len(L_sorted) * (1 - idk_ratio))
        indices_cur = list(indices[:test_num])
        y_truth_cur = [y_truth[i] for i in indices_cur]
        y_pred_cur = [y_pred[i] for i in indices_cur]

        human_indices = list(indices[test_num:])
        y_human = [y_truth[i] for i in human_indices]
        y_truth_cur = y_truth_cur + y_human
        y_pred_cur = y_pred_cur + y_human


        # added for same comparison to test process
        y_truth_cur = [ele.replace(' ,', ',') for ele in y_truth_cur]
        y_truth_cur = [ele.replace(' .', '.') for ele in y_truth_cur]

        y_pred_cur = [ele.replace(' ,', ',') for ele in y_pred_cur]
        y_pred_cur = [ele.replace(' .', '.') for ele in y_pred_cur]

        result = {}
        if 'Rouge' in metric_name :
            result = rouge_metric.compute(predictions=y_pred_cur, references=y_truth_cur, use_stemmer=True)
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        if 'BERTScore' in metric_name:
            bs_P, bs_R, bs_F = bert_metric(y_pred_cur, y_truth_cur)
            result["BERTScore_P"] = bs_P.mean().item()
            result["BERTScore_R"] = bs_R.mean().item()
            result["BERTScore_F"] = bs_F.mean().item()


        metric_score.append(result)

    save_file = os.path.join(res_save_path, res_save_name)
    if save_res == True:
        # save results
        metric_score_dump=json.dumps(metric_score)
        with open(save_file, 'w') as f:
            f.write(metric_score_dump)

        # save the res
        read_res_helper.read_force_eval_res(
            file_path=res_save_path,
            file_name=res_save_name,
            xlsx_save_name=xlsx_save_name
        )

    return metric_score, save_file


def cal_unc_oracle(gt_seq_list, pse_seq_list):
    pse_seq_list, gt_seq_list = postprocess_text(pse_seq_list, gt_seq_list)

    y_truth_cur = gt_seq_list
    y_pred_cur = pse_seq_list

    # added for same comparison to test process
    y_truth_cur = [ele.replace(' ,', ',') for ele in y_truth_cur]
    y_truth_cur = [ele.replace(' .', '.') for ele in y_truth_cur]

    y_pred_cur = [ele.replace(' ,', ',') for ele in y_pred_cur]
    y_pred_cur = [ele.replace(' .', '.') for ele in y_pred_cur]

    result = {}


    rouge_metric = load_metric("rouge")

    bertscorer = BERTScorer(lang="en", rescale_with_baseline=True)
    bert_metric = bertscorer.score

    # rouge
    result = rouge_metric.compute(predictions=y_pred_cur, references=y_truth_cur, use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # bertscore
    bs_P, bs_R, bs_F = bert_metric(y_pred_cur, y_truth_cur)
    result["BERTScore_P"] = bs_P
    result["BERTScore_R"] = bs_R
    result["BERTScore_F"] = bs_F

    print(result["BERTScore_F"].shape)
    res_score_bertscore = (1 - result["BERTScore_F"])


    return res_score_bertscore.tolist() 

def total_uncertainty(Baye_result):
    prob_all = []
    class_num = Baye_result[0].shape[1]
    for item in Baye_result:
        alpha = np.exp(item) + 1.0
        S = np.sum(alpha, axis=1, keepdims=True)
        prob = alpha / S
        prob_all.append(prob)
    prob_mean = np.mean(prob_all, axis=0)
    total_class_un = - prob_mean * (np.log(prob_mean) / np.log(class_num))
    total_un = np.sum(total_class_un, axis=1, keepdims=True)
    return total_un, total_class_un


def softmax(pred):
    ex = np.exp(pred - np.amax(pred, axis=1, keepdims=True))
    prob = ex / np.sum(ex, axis=1, keepdims=True)
    return prob


def entropy_softmax(pred):
    class_num = pred.shape[1]
    prob = softmax(pred) + 1e-10
    entropy = - prob * (np.log(prob) / np.log(class_num))
    total_un = np.sum(entropy, axis=1, keepdims=True)
    class_un = entropy
    return total_un, class_un


def aleatoric_uncertainty(Baye_result):
    al_un = []
    al_class_un = []
    for item in Baye_result:
        un, class_un = entropy_SL(item)
        al_un.append(un)
        al_class_un.append(class_un)
    ale_un = np.mean(al_un, axis=0)
    ale_class_un = np.mean(al_class_un, axis=0)
    return ale_un, ale_class_un


def entropy_SL(mean):
    class_num = mean.shape[1]
    alpha = np.exp(mean) + 1.0
    S = np.sum(alpha, axis=1, keepdims=True)
    prob = alpha / S
    entropy = - prob * (np.log(prob) / np.log(class_num))
    total_un = np.sum(entropy, axis=1, keepdims=True)
    class_un = entropy
    return total_un, class_un


def get_uncertainty(Baye_result):
    uncertainty = []
    uncertainty_class = []
    un_total, un_total_class = total_uncertainty(Baye_result)
    un_aleatoric, un_aleatoric_class = aleatoric_uncertainty(Baye_result)
    un_epistemic_class = un_total_class - un_aleatoric_class
    un_epistemic = np.sum(un_epistemic_class, axis=1, keepdims=True)

    uncertainty.append(un_aleatoric)
    uncertainty.append(un_epistemic)
    uncertainty.append(un_total)
    uncertainty_class.append(un_aleatoric_class)
    uncertainty_class.append(un_epistemic_class)
    uncertainty_class.append(un_total_class)
    return uncertainty

def entropy_dropout(pred):
    mean = []
    for p in pred:
        prob_i = softmax(p)
        mean.append(prob_i)
    mean = np.mean(mean, axis=0)
    class_num = mean.shape[1]
    prob = mean
    entropy = - prob * (np.log(prob) / np.log(class_num))
    total_un = np.sum(entropy, axis=1, keepdims=True)
    class_un = entropy
    return total_un, class_un


def aleatoric_dropout(Baye_result):
    al_un = []
    al_class_un = []
    for item in Baye_result:
        un, class_un = entropy_softmax(item)
        al_un.append(un)
        al_class_un.append(class_un)
    ale_un = np.mean(al_un, axis=0)
    ale_class_un = np.mean(al_class_un, axis=0)
    return ale_un, ale_class_un

def get_un_dropout(pred):
    un = []
    dr_entroy, dr_entroy_class = entropy_dropout(pred)
    dr_ale, dr_ale_clsss = aleatoric_dropout(pred)
    dr_eps_class = dr_entroy_class - dr_ale_clsss
    dr_eps = np.sum(dr_eps_class, axis=1, keepdims=True)
    un.append(dr_entroy)
    un.append(dr_ale)
    un.append(dr_eps)
    return un

def extract_bayes_res(all_score_list):
    round_num = len(all_score_list)
    sample_num = len(all_score_list[0])
    bayes_list = []
    for i in range(sample_num):
        mid_res = []
        for j in range(round_num):
            mid_res.append(all_score_list[j][i])
        bayes_list.append(mid_res)
    return bayes_list


def bnn_dp_cal_unc_score_list(all_cov_score_list, use_reciprocal):
    bayes_list = extract_bayes_res(all_cov_score_list)
    unc_res = []
    for bayes_ele in bayes_list:
        bayes_array = []
        for baye_ele in bayes_ele:
            baye_ele = np.array(baye_ele)
            baye_ele = np.expand_dims(baye_ele, axis=0)
            if use_reciprocal:

                baye_ele = 1 - baye_ele

            bayes_array.append(baye_ele)


        mid_res = get_un_dropout(bayes_array) # entropy, ale, eps
        print(f'mid_res={mid_res}')
        unc_res.append(mid_res)

    return unc_res

def transfer_to_multi_binary_label(sem_array):
    pos_array = np.expand_dims(sem_array, axis=-1)
    neg_array = 1 - pos_array
    res = np.concatenate((pos_array, neg_array), axis=-1)
    return res

def cal_min_max(list_list):
    min_val = float('inf')
    max_val = float('-inf')
    for ele in list_list:
        for midele in ele:
            for subele in midele:
                if subele < min_val:
                    min_val = subele
                if subele > max_val:
                    max_val = subele
    return min_val, max_val

def multilabel_bnn_dp_cal_unc_score_list(all_cov_score_list, use_reciprocal=True, mode="sum"):
    bayes_list = extract_bayes_res(all_cov_score_list)
    min_val, max_val = cal_min_max(bayes_list)
    dif_val = max_val - min_val + 1e-8
    unc_res = []
    for bayes_ele in bayes_list:
        sampled_prob = []
        for baye_ele in bayes_ele:
            baye_ele = np.array(baye_ele)
            baye_ele = (baye_ele - min_val) / dif_val


            if use_reciprocal:
                baye_ele = 1 - baye_ele
            binary_baye_ele = transfer_to_multi_binary_label(baye_ele)


            sampled_prob.append(binary_baye_ele)

        
        mid_res = get_un_dropout(sampled_prob) # entropy, ale, eps

        # to sum/mean mid_res
        mid_res_array = np.array(mid_res)
        if mode == 'mean':
            mid_mluti_unc = mid_res_array.mean(axis=1)
        elif mode == 'sum':
            mid_mluti_unc = mid_res_array.sum(axis=1)

        mid_mluti_unc = list(mid_mluti_unc.squeeze())

        # print(f'mid_res={mid_mluti_unc}')
        unc_res.append(mid_mluti_unc)

    return unc_res



def multilabel_bnn_var_cal(all_cov_score_list, mode="sum"):
    bayes_list = extract_bayes_res(all_cov_score_list)
    min_val, max_val = cal_min_max(bayes_list)
    dif_val = max_val - min_val + 1e-8
    unc_res = []
    for bayes_ele in bayes_list:

        bayes_ele = np.array(bayes_ele)
        bayes_ele = (bayes_ele - min_val) / dif_val

        # binary_baye_ele = transfer_to_multi_binary_label(baye_ele)
        bayes_ele_var = np.var(bayes_ele, axis=0)

        # to sum/mean mid_res
        if mode == 'mean':
            mid_mluti_unc = bayes_ele_var.mean()
        elif mode == 'sum':
            mid_mluti_unc = bayes_ele_var.sum()


        # print(f'mid_res={mid_mluti_unc}')
        unc_res.append(mid_mluti_unc)

    return unc_res



def get_mean_predct(bayes_array):
    try:
        bayes_array = np.array(bayes_array)
    except:
        print('invaid happends')
        return np.array([99])
    res = np.mean(bayes_array, axis=0)
    return res

def bnn_mean_cal_unc_score_list(all_cov_score_list):
    bayes_list = extract_bayes_res(all_cov_score_list)
    unc_res = [] # list of array
    wei_res = [] # lst of scalar
    for bayes_ele in bayes_list:
        bayes_array = []
        for baye_ele in bayes_ele:
            baye_ele = np.array(baye_ele)
            baye_ele = np.expand_dims(baye_ele, axis=0)

            bayes_array.append(baye_ele)
        mid_res = get_mean_predct(bayes_array) # mean of the sampling list

        unc_res.append(mid_res)

        wei_res.append(mid_res.mean().item())

    return unc_res, wei_res

def read_uncer_score_list(unc_res, index_enc=0):
    res = []
    for ele in unc_res:
        choose_score = ele[index_enc][0][0]
        res.append(choose_score)
    return res

def multilabel_read_uncer_score_list(unc_res, index_enc=0):
    res = []
    for ele in unc_res:
        print(ele)
        choose_score = ele[index_enc]
        res.append(choose_score)
    return res

