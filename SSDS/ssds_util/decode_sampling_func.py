# ******
# This is code is totally created by myself.
# ******
import numpy as np
import nltk
import json
import os
from bert_score import BERTScorer
from datasets import load_metric

class decode_sampling_helper(object):
    def __init__(self, preprocess_obj, use_parallel_sampling, parallel_sampling_id):
        self.preprocess_obj = preprocess_obj
        self.tokenizer=self.preprocess_obj.tokenizer
        self.bertscorer = BERTScorer(lang="en", rescale_with_baseline=True)
        self.metric = load_metric("rouge")
        if use_parallel_sampling == True:
            assert parallel_sampling_id is not None
        self.use_parallel_sampling = use_parallel_sampling
        self.parallel_sampling_id = parallel_sampling_id

    def decode_ids(self, ids):
        # ids: a tensor of text ids (n, max_dec_len)

        try:
            predictions = self.tokenizer.batch_decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
        except:
            print('************there are test decode error************')
            print(ids.shape)
            np.savetxt('./test_decode_error.txt', ids)
            print('save wrong info to ./decode_error.txt')
            # https://github.com/huggingface/transformers/issues/22634
            preds = np.where(ids != -100, ids, self.tokenizer.pad_token_id)
            predictions = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)





        return predictions

    def postprocess_text(self, texts):

        texts = [text.strip() for text in texts]

        texts = ["\n".join(nltk.sent_tokenize(text)) for text in texts]

        return texts

    def transfer_idlist_to_txtlist(self, pred_ids_list):
        pred_text_list_written = []
        pred_text_list_metric = []
        for pred_ids in pred_ids_list:
            mid_pred_text = self.decode_ids(pred_ids)
            mid_pred_text_written = [mid_pred.strip() for mid_pred in mid_pred_text]  # used for written
            mid_pred_text_metric = self.postprocess_text(mid_pred_text) # used for metrics
            pred_text_list_written.append(mid_pred_text_written)
            pred_text_list_metric.append(mid_pred_text_metric)
        return pred_text_list_written, pred_text_list_metric

    def list_to_json(self, pred_list, label_list, file_name):
        if os.path.isfile(file_name):
            os.remove(file_name)

        res = {}
        res['pred'] = pred_list
        res['label'] = label_list


        with open(file_name, 'w') as f:
            json_str = json.dumps(res)
            f.write(json_str)
        print(f"{file_name} has saved all sampled predictions ['pred'] and labels ['label'] .")

    def json_to_list(self, file_name):
        with open(file_name, 'r') as ini_f:
            ini_str = ini_f.readline()
            res = json.loads(ini_str) # res['pred'], res['label']
        return res

    def cal_bert_score(self, decoded_preds, decoded_labels):
        bs_P, bs_R, bs_F = self.bertscorer.score(decoded_preds, decoded_labels)

        BERTScore_P = bs_P.numpy()
        BERTScore_R = bs_R.numpy()
        BERTScore_F = bs_F.numpy()

        BERTScore_P = np.expand_dims(BERTScore_P, axis=0)
        BERTScore_R = np.expand_dims(BERTScore_R, axis=0)
        BERTScore_F = np.expand_dims(BERTScore_F, axis=0)

        return BERTScore_P, BERTScore_R, BERTScore_F


    def measure_pred_score(self, nlg_metric, decoded_preds, decoded_labels):
        used_labels = None
        if nlg_metric == 'oracle':
            used_labels = decoded_labels
            _, _, res = self.cal_bert_score(decoded_preds=decoded_preds, decoded_labels=used_labels)
        else:
            assert used_labels == None
        return res

    # (to do), add the original texts into measure_pred_score_list
    def measure_pred_score_list(self, nlg_metric, decoded_preds_list, decoded_labels_list=None):
        pred_score_list = []
        for i in range(len(decoded_preds_list)):
            mid_score_res = self.measure_pred_score(nlg_metric, decoded_preds_list[i], decoded_labels_list[i])
            pred_score_list.append(mid_score_res)

        print(f'each score_list has length of {pred_score_list[0].shape}')
        print(f"the total sampling time is {len(pred_score_list)}")

        pred_score_array = np.concatenate(pred_score_list, axis=0)

        return pred_score_array


    def score_decoding_sampling(self, pred_ids_list, label_ids_list, pred_txt_list_save_path, nlg_metric):
        # pred_ids_list: a list of tensor, where each tensor is prediction ids [(n, max_dec_len), ..., (n, max_dec_len)]
        # label_ids_list: a list of tensor, where each tensor is label ids [(n, label_len), ..., (n, label_len)]
        # pred_txt_list_save_path: a json file path to save the texts of pred_ids_list and label_ids_list
            # key in json file of pred_txt_list_save_path:

        sampling_times = len(pred_ids_list)
        assert len(label_ids_list) == 1 or len(label_ids_list) == sampling_times
        if label_ids_list == 1:
            label_ids_list = [label_ids_list[0]] * sampling_times

        # step a. (optional) transfer to a list of texts and save
        pred_text_list_written, pred_text_list_metric = self.transfer_idlist_to_txtlist(pred_ids_list=pred_ids_list)
        label_text_list_written, label_text_list_metric = self.transfer_idlist_to_txtlist(pred_ids_list=label_ids_list)
        if self.use_parallel_sampling == True:
            self.list_to_json(pred_list=pred_text_list_written, label_list=label_text_list_written,
                              file_name=pred_txt_list_save_path[:-5] + str(self.parallel_sampling_id) + '.json')
        else:
            self.list_to_json(pred_list=pred_text_list_written, label_list=label_text_list_written, file_name=pred_txt_list_save_path)
        

        # step b. use a function to cal_score
        if nlg_metric == 'oracle':
            pred_score_mat = self.measure_pred_score_list(
                nlg_metric=nlg_metric,
                decoded_preds_list=pred_text_list_metric,
                decoded_labels_list=label_text_list_metric,
            )
        elif nlg_metric == None:
            pred_score_mat = None
        else:
            raise ValueError(f'the nlg_metric={nlg_metric} is wrongly set!')

        # save pred_score_mat
        if nlg_metric != None:
            np.save(pred_txt_list_save_path[:-5] + '_score_array.npy', pred_score_mat)
            print(f'save pred_score_mat to {pred_txt_list_save_path[:-5]}_score_array.npy')

            best_index = np.argmax(pred_score_mat, axis=0)
            best_score_1d = np.max(pred_score_mat, axis=0)
            best_sampled_pred = []
            best_sampled_label = []
            best_sampled_txtid = []
            for i in range(best_index.shape[0]):
                best_sampled_pred.append(pred_text_list_metric[best_index[i]][i])
                best_sampled_label.append(label_text_list_metric[best_index[i]][i])
                best_sampled_txtid.append(pred_ids_list[best_index[i]][i])
        else:
            best_sampled_pred = pred_text_list_metric[0]
            best_sampled_label = label_text_list_metric[0]
            best_sampled_txtid = pred_ids_list[0]
            best_score_1d = None

        return best_sampled_txtid, best_sampled_pred, best_sampled_label, best_score_1d, pred_score_mat

    # step c. filter_out/weigh according to the cal_score
    def filter_weigh_on_scores(self, input_txtids, input_preds, input_labels, input_scores, filter_mode, filter_thrs):
        filtered_sampled_pred = []
        filtered_sampled_label = []
        filtered_sampled_id = []
        filtered_sampled_txtid = []

        if filter_mode == None:
            return input_txtids, input_preds, input_labels, None
        elif 'select' in filter_mode and 'weigh' not in filter_mode:
            assert len(input_scores.shape) == 1
            for i in range(input_scores.shape[0]):
                if input_scores[i] > filter_thrs:
                    filtered_sampled_txtid.append(input_txtids[i])
                    filtered_sampled_pred.append(input_preds[i])
                    filtered_sampled_label.append(input_labels[i])
                    filtered_sampled_id.append(i)
        else:
            raise ValueError(f"filter_mode={filter_mode} is wrongly set")
        return filtered_sampled_txtid, filtered_sampled_pred, filtered_sampled_label, filtered_sampled_id

    # step d. return filter_out/weigh samples

    def compute_metrics(self, pred_ids, decoded_preds, decoded_labels):


        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}


        # token-level length used in metric_util.py -> compute_metrics_wBERTscore
        prediction_lens = [np.count_nonzero(pred_id != self.tokenizer.pad_token_id) for pred_id in pred_ids]

        # char-level length
        prediction_lens = np.array(prediction_lens)
        result["gen_len"] = np.mean(prediction_lens)

        print(f"decoded_preds len is {len(decoded_preds)}")
        print(f"decoded_labels len is {len(decoded_labels)}")

        bs_P, bs_R, bs_F = self.bertscorer.score(decoded_preds, decoded_labels)


        result["BERTScore_P"] = bs_P.mean().item()
        result["BERTScore_R"] = bs_R.mean().item()
        result["BERTScore_F"] = bs_F.mean().item()

        result = {k: round(v, 4) for k, v in result.items()}
        return result
