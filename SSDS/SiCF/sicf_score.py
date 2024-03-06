import copy

#### this is a general NLG uncertainty estimation package####
# it includes the below
# 1. semantic invariance (positive impact to NLG prediction)
# 2. coverage (positive impact to NLG prediction)
# 3. faithfulness (positive impact to NLG prediction)
# 4. BNN-based uncertainty metrics
#### the data type is mostly based on numpy for general usage ####

import numpy as np
from scipy.spatial import distance
from tqdm import tqdm
import torch.nn as nn
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from flair.data import Sentence
from flair.models import SequenceTagger
from .nertagger import NERTagger
from .postagger import POSTagger

from sentence_transformers import SentenceTransformer
from collections import Counter

import pickle
import torch

from operator import itemgetter

class SiCF_Tool(object):
    def __init__(self, save_path,
                 use_cov_count=True,
                 use_fai_count=True,
                 use_small=False,
                 prefix='',
                 fai_cal_type='nli',
                 sein_dis='euclidean',
                 cov_dis='euclidean',
                 fai_dis='euclidean'):
        self.save_path=save_path
        self.use_cov_count=use_cov_count
        self.use_fai_count=use_fai_count
        self.use_small=use_small
        self.prefix=prefix
        self.fai_cal_type=fai_cal_type
        self.sein_dis=sein_dis  # if set 'cosine', choose beset from min to max
        self.cov_dis=cov_dis # if set 'cosine', choose beset from min to max
        self.fai_dis=fai_dis # if set 'cosine', choose beset from min to max


    def emb_semantic_invariance(self, emb_array):
        # emb_array: a m-length list of embed array-(b * d) for b samples with d-dimension emb

        # sem_inv_mic: b * d arraies, where each d-dim array shows semantic invarance for a sample in a micro-level
        # sem_inv_pool: b-length list of scalars, each scalar shows semantic invarance for a sample
        # sem_dist_arr: b * m arraies, where each row shows the distance from m samples to the
        # sem_disrank_list: b-length list of scalars, each scalar shows semantic invarance for a sample

        # emb_array = np.array(emb_list) # m * b * d
        m, b, d = emb_array.shape

        sem_inv_mic = []
        sem_inv_pool = []
        sem_dist_arr = []
        sem_disrank_list = []

        for i in range(b):
            single_emb_array = emb_array[:, i, :] # m * d
            single_emb_var_d = np.var(single_emb_array, axis=0) # d-dim: variance of each dimension emb in the d-dimension vector
            single_emb_var_1 = np.mean(single_emb_var_d) # 1-dim

            # look for the min/max distances
            single_emb_mean_d = np.mean(single_emb_array, axis=0)
            single_emb_mean_d = np.expand_dims(single_emb_mean_d, axis=0)
            dist_d = distance.cdist(single_emb_mean_d, single_emb_array, self.sein_dis)
            if self.sein_dis in ['euclidean']:
                min_dist = np.min(dist_d, axis=1)
            elif self.sein_dis in ['cosine']:
                min_dist = np.min(dist_d, axis=1)
            else:
                raise ValueError(f'SiCF_too.sein_dis={self.sein_dis} is wrongly set')
            min_index = np.where(dist_d==min_dist)
            min_index = min_index[1][0]

            sem_dist_arr.append(dist_d)
            sem_disrank_list.append(min_index)



            sem_inv_mic.append(np.expand_dims(single_emb_var_d, axis=0))
            sem_inv_pool.append(single_emb_var_1)



        sem_inv_mic = np.concatenate(sem_inv_mic, axis=0) # b * d
        sem_dist_arr = np.concatenate(sem_dist_arr, axis=0) # b * m

        print('finished semantic invariance calclation')
        return sem_inv_mic, sem_inv_pool, sem_dist_arr, sem_disrank_list

    def txt_semantic_invariance(self, txt_list, cal_sein_emb=True, batch_size=128, model_name=None):

        if self.use_small:
            sein_emb_path = os.path.join(self.save_path, self.prefix+'_'+'toy_sein_emb_array.npy')
        else:
            sein_emb_path = os.path.join(self.save_path, self.prefix+'_'+'full_sein_emb_array.npy')
        print(f"sein_emb_path is {sein_emb_path}")

        if cal_sein_emb:
            if os.path.isfile(sein_emb_path) and not self.use_small:
                raise ValueError(f'{sein_emb_path} is existing! You can only regenerate one by manually delete the existing one.')
            from transformers import RobertaTokenizer, RobertaModel
            tokenizer = RobertaTokenizer.from_pretrained(model_name)

            # bert = RobertaModel.from_pretrained(model_name).cuda() # single gpu
            bert = RobertaModel.from_pretrained(model_name)
            bert = nn.DataParallel(bert)
            bert = bert.cuda()
            # transfer to emb_list (to do, chage into batch size format)
            emb_list = []
            for texts in txt_list:
                mid_emb_list = []
                # iterator = range(0, len(texts), batch_size)
                iterator = tqdm(range(0, len(texts), batch_size), desc="Semantic Invariance")
                for batch_idx in iterator:
                    text = texts[batch_idx:batch_idx+batch_size]
                # for text in texts:
                    input = tokenizer(text, padding=True, return_tensors="pt")
                    # input_mask = input['attention_mask'].cuda()
                    # input_id = input['input_ids'].cuda()
                    for key, value in input.items():
                        input[key] = input[key].cuda()
                    output = bert(**input)
                    mid_emb_list.append(output[1].detach().cpu().numpy())
                mid_emb_array = np.concatenate(mid_emb_list, axis=0)

                emb_list.append(mid_emb_array)

            emb_array = np.array(emb_list)
            print(f"emb_array has shape of {emb_array.shape}")
            np.save(sein_emb_path, emb_array)
        else:
            emb_array = np.load(sein_emb_path)

        sem_inv_mic, sem_inv_pool, sem_dist_arr, sem_disrank_list = self.emb_semantic_invariance(emb_array)
        return sem_inv_mic, sem_inv_pool, sem_dist_arr, sem_disrank_list

    def extract_entities(self, src_phrase):
        phrase = src_phrase[0]
        phrase = list(phrase)
        occurances = Counter(phrase)

        return occurances

    def select_extract_entities(self, src_phrase):
        phrase = src_phrase[0]
        phrase = list(phrase)

        type = src_phrase[1]

        src_dict = {}

        for i in range(len(phrase)):
            if phrase[i] in src_dict.keys() and 'NNP' in src_dict[phrase[i]]:
                continue
            src_dict[phrase[i]] = type[i]

        occurances = Counter(phrase)
        for ele in occurances.keys():
            if 'NNP' in src_dict[ele]:
                occurances[ele] = 1


        return occurances

    def extract_keyword_list(self, occurances):
        return list(occurances.keys())

    def counts_keyword(self, occurances, keyword_list):
        res = []
        for ele in keyword_list:
            res.append(occurances[ele])
        return res

    def all_zero_dict(self, src_occurance):
        dict_src_occurance = dict(src_occurance)
        for key in dict_src_occurance.keys():
            if dict_src_occurance[key] != 0:
                return False
        return True

    def moved_sigmoid(self, x):
        x = x - 1
        s = 1 / (1 + np.exp(-x))
        return s

    def process_fai_count(self, ref_occurance_entity_count):
        count_array = np.array(ref_occurance_entity_count)
        pro_score = self.moved_sigmoid(count_array)
        return list(pro_score)

    def process_cov_count(self, ref_occurance_entity_count):
        count_array = np.array(ref_occurance_entity_count)
        pro_score = self.moved_sigmoid(count_array)
        return list(pro_score)


    def inter_cal_cov_onesample(self, src, ref, phrase_model, start, end, max_dict_len, no_cov_score=None):
         # Return:
         # src_score_list:  a list of softmax probability / an array with padding in different lengths
        if self.cov_dis in ['euclidean']:
            no_cov_score = 50
        elif self.cov_dis in ['cosine']:
            no_cov_score = 50

        score_list = []

        index = list(range(start, end))
        # assert len(src) == len(ref)
        if self.use_small:
            print(f"src has l={len(src)}, ref has l={len(ref)}")
            print(f"max_dict_len={max_dict_len}")
        # if len(src) != len(ref):
        #     print('stop here')
        # for i in range(start, end):
        for i in range(len(src)):
            if self.use_small:
                print(index[i])
                print(f'i={i}')
                print(f'index[i]={index[i]}')
            if index[i] >= max_dict_len:
                break

            src_phrase = src[index[i]]
            ref_phrase = ref[index[i]]
            if self.use_small:
                print(src_phrase)
                print(ref_phrase)


            src_occurance = self.select_extract_entities(src_phrase)
            ref_occurance = self.select_extract_entities(ref_phrase)

            if len(src_occurance) != 0 and len(ref_occurance) != 0 and not self.all_zero_dict(ref_occurance):
            # get the entity from the dict
                src_occurance_eneity_list = self.extract_keyword_list(src_occurance)
                ref_occurance_entity_list = self.extract_keyword_list(ref_occurance)
                if self.use_small:
                    print(ref_occurance_entity_list)

                ref_occurance_entity_count = self.counts_keyword(ref_occurance, ref_occurance_entity_list)

                # get an array of entity embeddings
                src_emb = phrase_model.encode(src_occurance_eneity_list)
                ref_emb = phrase_model.encode(ref_occurance_entity_list)

                # get this round distance between src and ref
                dist_d = distance.cdist(ref_emb, src_emb, self.cov_dis)
                if self.cov_dis in ['euclidean']:
                    min_dist = np.min(dist_d, axis=1)
                elif self.cov_dis in ['cosine']:
                    min_dist = np.min(dist_d, axis=1)
                else:
                    raise ValueError(f'SiCF_too.cov_dis={self.cov_dis} is wrongly set')

                if self.use_cov_count:
                    processsed_ref_occurance_entity_count = self.process_cov_count(ref_occurance_entity_count)
                    min_dist = min_dist * processsed_ref_occurance_entity_count # need double check, show to others whether use padding in column
                min_dist = list(min_dist)

            else:
                # min_dist = no_cov_score
                if len(ref_occurance) != 0:
                    min_dist = [no_cov_score] * len(ref_occurance)
                else:
                    min_dist = [no_cov_score] * 1

            score_list.append(min_dist)
        return score_list


    def check_txt_coverage_quality(self, src_list, ref, gt, cal_cov_emb=True, batch_size=4, model_name=None):
        # cov_emb_path = os.path.join(self.save_path, 'cov_emb_array.npy')

        if self.use_small:
            cov_emb_path = os.path.join(self.save_path, self.prefix+'_'+'toy_cov_emb_pickle.txt')
        else:
            cov_emb_path = os.path.join(self.save_path, self.prefix+'_'+'full_cov_emb_pickle.txt')


        # cov_emb_path = os.path.join(self.save_path, 'cov_emb_pickle.txt')
        print(f"cov_emb_path is {cov_emb_path}")

        # ner_tagger = NERTagger(tagger_id="flair/ner-english-ontonotes")
        ner_tagger = POSTagger(tagger_id="flair/pos-english")
        phrase_model = SentenceTransformer('whaleloops/phrase-bert')


        if cal_cov_emb:
            if os.path.isfile(cov_emb_path) and not self.use_small:
                raise ValueError(f'{cov_emb_path} is existing! You can only regenerate one by manually deleting the existing one.')
            all_cov_score_list = [] # a list of each round-level cov score list
            cal_ref_ner = True
            full_ref_tagged_dict = {}
            for summaries in src_list:
                round_cov_score_list = []
                # iterator = range(0, len(texts), batch_size)
                iterator = tqdm(range(0, len(summaries), batch_size), desc="Checking Coverage")
                for batch_idx in iterator:
                    batch_summary = summaries[batch_idx:batch_idx + batch_size]
                    summary_tagged_dict = ner_tagger.batch_tag(batch_summary, batch_idx, batch_idx + batch_size)
                    print("pd_sum info is below")
                    print(batch_summary)
                    print(summary_tagged_dict)


                    if cal_ref_ner:
                        batch_ref = ref[batch_idx:batch_idx + batch_size]
                        ref_tagged_dict = ner_tagger.batch_tag(batch_ref, batch_idx, batch_idx + batch_size)
                        print("ori_dialogue info is below")
                        print(batch_ref)
                        print(ref_tagged_dict)
                        batch_gt = gt[batch_idx:batch_idx + batch_size]
                        gt_tagged_dict = ner_tagger.batch_tag(batch_gt, batch_idx, batch_idx + batch_size)
                        print("gt_sum info is below")
                        print(batch_gt)
                        print(gt_tagged_dict)

                        print('\n')




    def txt_coverage(self, src_list, ref, cal_cov_emb=True, batch_size=4, model_name=None):
        # cov_emb_path = os.path.join(self.save_path, 'cov_emb_array.npy')

        if self.use_small:
            cov_emb_path = os.path.join(self.save_path, self.prefix+'_'+'toy_cov_emb_pickle.txt')
        else:
            cov_emb_path = os.path.join(self.save_path, self.prefix+'_'+'full_cov_emb_pickle.txt')


        # cov_emb_path = os.path.join(self.save_path, 'cov_emb_pickle.txt')
        print(f"cov_emb_path is {cov_emb_path}")

        ner_tagger = POSTagger(tagger_id="flair/pos-english")
        phrase_model = SentenceTransformer('whaleloops/phrase-bert')


        if cal_cov_emb:
            if os.path.isfile(cov_emb_path) and not self.use_small:
                raise ValueError(f'{cov_emb_path} is existing! You can only regenerate one by manually deleting the existing one.')
            all_cov_score_list = [] # a list of each round-level cov score list
            cal_ref_ner = True
            full_ref_tagged_dict = {}
            for summaries in src_list:
                round_cov_score_list = []

                iterator = tqdm(range(0, len(summaries), batch_size), desc="Coverage")
                for batch_idx in iterator:
                    batch_summary = summaries[batch_idx:batch_idx + batch_size]
                    summary_tagged_dict = ner_tagger.batch_tag(batch_summary, batch_idx, batch_idx + batch_size)

                    if cal_ref_ner:
                        batch_ref = ref[batch_idx:batch_idx + batch_size]
                        ref_tagged_dict = ner_tagger.batch_tag(batch_ref, batch_idx, batch_idx + batch_size)
                        for key, val in ref_tagged_dict.items():
                            full_ref_tagged_dict[key] = val
                    max_dict_len = len(full_ref_tagged_dict)
                    # cal the batch level cov
                    score_list = self.inter_cal_cov_onesample(src=summary_tagged_dict,
                                                 ref=full_ref_tagged_dict,
                                                 phrase_model=phrase_model,
                                                 start=batch_idx,
                                                 end=batch_idx + batch_size,
                                                 max_dict_len=max_dict_len,
                                                 )
                    # save in the batch level
                    round_cov_score_list.extend(score_list)
                    # print(round_cov_score_list)
                cal_ref_ner = False
                # save in the round level
                all_cov_score_list.append(round_cov_score_list)

            print(f"all_cov_score_list has len of {len(all_cov_score_list)}")
            # if self.use_small:
            #     print(f"{len(all_cov_score_list[0])} and {len(all_cov_score_list[1])} and {len(all_cov_score_list[2])} ")

            with open(cov_emb_path, "wb") as fp:  # Pickling
                pickle.dump(all_cov_score_list, fp)
        else:
            # all_cov_score_list = np.load(cov_emb_path)
            with open(cov_emb_path, "rb") as fp:  # Unpickling
                all_cov_score_list = pickle.load(fp)
        return all_cov_score_list



    def faithfulness(self, src_list, ref, dis=None):
        if dis == None:
            dis = self.fai_dis
        pass

    def sicf_score(self):
        pass

    def get_total_rank(self, wei):
        if not type(wei) is np.ndarray:
            return 0
        total_len = wei.shape[0]
        return total_len

    def get_total_coef(self, wei, coef):
        if not type(wei) is np.ndarray:
            return 0
        return coef

    def normalize_wei(self, wei):
        if not type(wei) is np.ndarray:
            return 0
        wei_min = np.min(wei).item()
        wei_max = np.max(wei).item() + 1e-8
        return (wei - wei_min)/(wei_max - wei_min)


    def wei2rank(self, wei):
        if not type(wei) is np.ndarray:
            return 0
        indices, L_sorted = zip(*sorted(enumerate(wei), key=itemgetter(1), reverse=True))

        len_wei = len(wei)
        dict_wei2rank = {}
        for i in range(len(indices)):
            dict_wei2rank[str(indices[i])] = len_wei - i - 1
        rank = [-1] * len_wei
        for i in range(len(indices)):
            rank[int(indices[i])] = dict_wei2rank[str(indices[i])]
        return np.array(rank)

    def process_wei_occur(self, min_dist0, ref_occurance_entity_count0, special_value=50):
        if 0 not in ref_occurance_entity_count0:
            return min_dist0, ref_occurance_entity_count0
        min_dist = copy.deepcopy(min_dist0)
        ref_occurance_entity_count = copy.deepcopy(ref_occurance_entity_count0)
        assert min_dist.shape[0] == len(ref_occurance_entity_count)
        for i in range(len(ref_occurance_entity_count)):
            if ref_occurance_entity_count[i]==0:
                min_dist[i] = special_value
                ref_occurance_entity_count[i] = 1
        return min_dist, ref_occurance_entity_count

    def intersec_concate_two_list(self, list_a, list_b):
        res = []
        for a in list_a:
            for b in list_b:
                mid_res = (a, b)
                res.append(mid_res)
        return res

    def intersec_concate_two_list_ori(self, list_a, list_b):
        res = []
        for a in list_a:
            for b in list_b:
                mid_res = (a, b)
                res.append(mid_res)
        return res

    def get_nli_socre(self, gt_sent, pd_sent, tokenizer, model):
        concate_sent_list = self.intersec_concate_two_list(gt_sent, pd_sent)
        print(f"len(concate_sent_list)={len(concate_sent_list)}")

        batch_tokens = tokenizer.batch_encode_plus(concate_sent_list, padding=True,
                                                        truncation=True, max_length=512,
                                                        return_tensors="pt", truncation_strategy="only_first")
        with torch.no_grad():
            model_outputs = model(**{k: v.cuda() for k, v in batch_tokens.items()})

        batch_probs = torch.nn.functional.softmax(model_outputs["logits"], dim=-1)
        batch_evids = batch_probs[:, self.entailment_idx]
        batch_conts = batch_probs[:, self.contradiction_idx]
        # batch_neuts = batch_probs[:, self.neutral_idx].tolist()

        res =  batch_conts - batch_evids # a smaller one means better faithfulness, same direction to Euclidean distance.

        dist_d = res.reshape(len(gt_sent), len(pd_sent))

        dist_d = dist_d.cpu().numpy()

        min_dist = np.min(dist_d, axis=1)

        return min_dist

    def inter_cal_fai_onesample(self, src, ref, sent_toknizer, sent_model, start, end, max_dict_len, no_fai_score=None):
         # Return:
         # src_score_list:  a list of softmax probability / an array with padding in different lengths
        if self.cov_dis in ['euclidean']:
            no_fai_score = 50
        elif self.cov_dis in ['cosine']:
            no_fai_score = 50

        score_list = []

        index = list(range(start, end))
        # assert len(src) == len(ref)
        if self.use_small:
            print(f"src has l={len(src)}, ref has l={len(ref)}")
            print(f"max_dict_len={max_dict_len}")

        for i in range(len(src)):
            if self.use_small:
                print(index[i])
                print(f'i={i}')
                print(f'index[i]={index[i]}')
            if index[i] >= max_dict_len:
                break

            src_phrase = src[index[i]]
            ref_phrase = ref[index[i]]
            if self.use_small:
                print(src_phrase)
                print(ref_phrase)

            # get a dict of {entity: occurance} # in fai: entity -> sent
            src_occurance = src_phrase
            ref_occurance = ref_phrase

            if len(src_occurance) != 0 and len(ref_occurance) != 0 and not self.all_zero_dict(ref_occurance):
            # get the entity from the dict
                src_occurance_eneity_list = self.extract_keyword_list(src_occurance)
                ref_occurance_entity_list = self.extract_keyword_list(ref_occurance)
                if self.use_small:
                    print(ref_occurance_entity_list)

                # get the entity occurance from the dict
                ref_occurance_entity_count = self.counts_keyword(ref_occurance, ref_occurance_entity_list)



                # get this round distance between src and ref
                if self.fai_cal_type=='sim':
                    # get an array of entity embeddings
                    src_emb = sent_model.encode(src_occurance_eneity_list)
                    ref_emb = sent_model.encode(ref_occurance_entity_list)

                    # cal similarity
                    dist_d = distance.cdist(ref_emb, src_emb, self.cov_dis)
                    if self.fai_dis in ['euclidean']:
                        min_dist = np.min(dist_d, axis=1)
                    elif self.fai_dis in ['cosine']:
                        min_dist = np.min(dist_d, axis=1)
                    else:
                        raise ValueError(f'SiCF_too.cov_dis={self.cov_dis} is wrongly set')

                elif self.fai_cal_type=='nli':
                    min_dist = self.get_nli_socre(
                        gt_sent=ref_occurance_entity_list,
                        pd_sent=src_occurance_eneity_list,
                        tokenizer=sent_toknizer,
                        model=sent_model
                    )

                if self.use_fai_count:
                    processed_ref_occurance_entity_count = self.process_fai_count(ref_occurance_entity_count)
                    min_dist = min_dist * processed_ref_occurance_entity_count # need double check, show to others whether use padding in column

                min_dist = list(min_dist)

            else:
                # min_dist = no_fai_score
                if len(ref_occurance) != 0:
                    min_dist = [no_fai_score] * len(ref_occurance)
                else:
                    min_dist = [no_fai_score] * 1

            score_list.append(min_dist)
        return score_list



    def txt_faithfulness(self, src_list, ref, cal_fai_emb=True, batch_size=4, model_name=None):
        if self.use_small:
            fai_emb_path = os.path.join(self.save_path, self.prefix+'_'+'toy_fai_emb_pickle.txt')
        else:
            fai_emb_path = os.path.join(self.save_path, self.prefix+'_'+'full_fai_emb_pickle.txt')


        print(f"fai_emb_path is {fai_emb_path}")

        # ner_tagger = NERTagger(tagger_id="flair/ner-english-ontonotes")
        ner_tagger = POSTagger(tagger_id="flair/pos-english")
        if self.fai_cal_type == 'sim':
            sent_toknizer = None
            sent_model = SentenceTransformer('deepset/sentence_bert')
        elif self.fai_cal_type == 'nli':
            model_card = 'microsoft/deberta-base-mnli'
            # model_card = 'tals/albert-xlarge-vitaminc-mnli'
            sent_toknizer = AutoTokenizer.from_pretrained(model_card)
            sent_model = AutoModelForSequenceClassification.from_pretrained(model_card).eval()
            sent_model = sent_model.cuda().half()

            if model_card == 'microsoft/deberta-base-mnli':
                self.entailment_idx = 2
                self.contradiction_idx = 0
            elif model_card == 'tals/albert-xlarge-vitaminc-mnli':
                self.entailment_idx = 0
                self.contradiction_idx = 1

            else:
                raise ValueError(f"model_card={model_card} is wrongly set.")

        if cal_fai_emb:
            if os.path.isfile(fai_emb_path) and not self.use_small:
                raise ValueError(f'{fai_emb_path} is existing! You can only regenerate one by manually delete the existing one.')
            all_fai_score_list = [] # a list of each round-level cov score list
            cal_ref_ner = True
            full_ref_tagged_dict = {}
            for summaries in src_list:
                round_fai_score_list = []
                # iterator = range(0, len(texts), batch_size)
                iterator = tqdm(range(0, len(summaries), batch_size), desc="Faithfulness")
                for batch_idx in iterator:
                    batch_summary = summaries[batch_idx:batch_idx + batch_size]
                    summary_tagged_dict = ner_tagger.batch_find_keysent(batch_summary, batch_idx, batch_idx + batch_size)
                    # list(summary_tagged_dict[0].keys())[0].text # if output key is the sentence->span

                    if cal_ref_ner:
                        batch_ref = ref[batch_idx:batch_idx + batch_size]
                        ref_tagged_dict = ner_tagger.batch_find_keysent(batch_ref, batch_idx, batch_idx + batch_size)
                        for key, val in ref_tagged_dict.items():
                            full_ref_tagged_dict[key] = val
                    max_dict_len = len(full_ref_tagged_dict)
                    # cal the batch level fai
                    score_list = self.inter_cal_fai_onesample(src=summary_tagged_dict,
                                                 ref=full_ref_tagged_dict,
                                                 sent_toknizer=sent_toknizer,
                                                 sent_model=sent_model,
                                                 start=batch_idx,
                                                 end=batch_idx + batch_size,
                                                 max_dict_len=max_dict_len,
                                                 )
                    # save in the batch level
                    round_fai_score_list.extend(score_list)
                    # print(round_fai_score_list)
                cal_ref_ner = False
                # save in the round level
                all_fai_score_list.append(round_fai_score_list)
            print(f"all_fai_score_list has len of {len(all_fai_score_list)}")
            # if self.use_small:
            #     print(f"{len(all_fai_score_list[0])} and {len(all_fai_score_list[1])} and {len(all_fai_score_list[2])} ")

            with open(fai_emb_path, "wb") as fp:  # Pickling
                pickle.dump(all_fai_score_list, fp)
        else:
            # all_fai_score_list = np.load(fai_emb_path)
            with open(fai_emb_path, "rb") as fp:  # Unpickling
                all_fai_score_list = pickle.load(fp)
        return all_fai_score_list


    def preprocess_text(self, text_list):
        res = [ele.replace("\n", " ") for ele in text_list]
        return res