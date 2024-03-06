from dataclasses import dataclass

from flair.data import Sentence
from flair.models import SequenceTagger
from collections import Counter

import nltk

@dataclass
class POSTagger:
    """NER tagger based on flair package"""

    tagger_id: str = "flair/pos-english" 
    # https://huggingface.co/flair/pos-english

    def __post_init__(self):
        self.tagger = SequenceTagger.load(self.tagger_id)

    def get_one_sent_tag_res(self, sentence):
        m = sentence.to_tagged_string()
        sent1 = m.split("→ [")[1][:-1]
        tokens = [x.split("/")[0].replace('"', '') for x in sent1.split()]  # token text
        poses = [x.split("/")[1] for x in sent1.split()]
        poses_0 = [x[:-1] for x in poses[:-1]] + [poses[-1]]  #

        pos_list = []
        score_list = []
        for entity in sentence.get_labels('pos'):
            pos_list.append(entity.value)
            score_list.append(entity.score)

        slot_values = tuple(tokens)
        slot_types = tuple(poses_0)
        slot_socres = tuple(score_list)
        return slot_values, slot_types, slot_socres



    def tag(self, sentences):
        # output format: a map of sentence -> (slot_values, slot_types)
        sentences = [Sentence(text) for text in sentences]
        self.tagger.predict(sentences)

        tagged_dict = {}
        for sentence in sentences:
            slot_values, slot_types, slot_socres = self.get_one_sent_tag_res(sentence=sentence)
            slot_types, slot_values, slot_socres = self.select_pos_type(type_list=slot_types,
                                                                        text_list=slot_values,
                                                                        score_list=slot_socres)
            tagged_dict[sentence.text] = (slot_values, slot_types)


        return tagged_dict

    def calculate_tags(self, asent):
        sentence = Sentence(asent)
        self.tagger.predict(sentence)
        m = sentence.to_tagged_string()
        sent1 = m.split("→ [")[1][:-1]
        tokens = [x.split("/")[0].replace('"', '') for x in sent1.split()]
        poses = [x.split("/")[1] for x in sent1.split()]
        poses_0 = [x[:-1] for x in poses[:-1]] + [poses[-1]]
        atags = list(zip(tokens, poses_0))
        return atags

    def batch_tag(self, sentences, start, end):
        # output format: a map of sentence -> (slot_values, slot_types)
        sentences = [Sentence(text) for text in sentences]
        self.tagger.predict(sentences)

        tagged_dict = {}
        sent_len = len(sentences)
        index = list(range(start, end))
        for i in range(len(sentences)):
            sent = sentences[i]
            slot_values, slot_types, slot_socres = self.get_one_sent_tag_res(sentence=sent)
            slot_types, slot_values, slot_socres = self.select_pos_type(type_list=slot_types,
                                                                        text_list=slot_values,
                                                                        score_list=slot_socres)
            tagged_dict[index[i]] = (slot_values, slot_types)

        return tagged_dict

    def select_pos_type(self, type_list, text_list, score_list, type_kw='NN'):
        new_type_list = []
        new_text_list = []
        new_score_list = []
        for i in range(len(type_list)):
            if type_kw in type_list[i]:
                new_type_list.append(type_list[i])
                new_text_list.append(text_list[i])
                new_score_list.append(score_list[i])

        new_type_list = tuple(new_type_list)
        new_text_list = tuple(new_text_list)
        new_score_list = tuple(new_score_list)

        return new_type_list, new_text_list, new_score_list

    def chunk_long_sent(self, sent_list, max_chunk_num):
        tar_sent_list = []
        ori_sent_len = len(sent_list)
        each_chunk_len = int(ori_sent_len/max_chunk_num)
        remain_chunk_len = ori_sent_len % max_chunk_num
        for i in range(max_chunk_num):
            mid_res = []
            if i != max_chunk_num - 1 or remain_chunk_len == 0:
                mid_res.extend(sent_list[(i * each_chunk_len) : ((i+1) * each_chunk_len)])
            else:
                mid_res.extend(sent_list[(i * each_chunk_len): (i * each_chunk_len + remain_chunk_len)])
            ini_str = ' '
            merg_str = ini_str.join(mid_res)
            tar_sent_list.append(merg_str)
        return tar_sent_list




    def batch_find_keysent(self, sentences, start, end, overlooks_NNP=True, max_chunk_num=20):
        # output format: a map of sentence -> (slot_values, slot_types)
        # sentences = [Sentence(text) for text in sentences]
        # self.tagger.predict(sentences)


        tagged_dict = {}
        sent_len = len(sentences)
        index = list(range(start, end))
        for i in range(len(sentences)):
            sent = sentences[i]
            subsent_list = nltk.sent_tokenize(sent)

            subsent_list = [Sentence(text) for text in subsent_list]
            self.tagger.predict(subsent_list)
            res = {}
            for subsent in subsent_list:
                slot_values, slot_types, slot_socres = self.get_one_sent_tag_res(subsent)
                slot_types, slot_values, slot_socres = self.select_pos_type(type_list=slot_types,
                                                                            text_list=slot_values,
                                                                            score_list=slot_socres)
                count = 0
                occurances = Counter(slot_types)
                if overlooks_NNP == True:
                    # then start ignore NNP

                    for kw in occurances.keys():
                        if 'NNP' not in kw:
                            count += occurances[kw]
                        # else:
                        #     count += 1 # maximum has 1 for many NNP or NNPS cases
                else:
                    for kw in occurances.keys():
                        count += occurances[kw]

                res[subsent.text] = count
            tagged_dict[index[i]] = res
        return tagged_dict

    @staticmethod
    def block_slot_types(tagged_dict, block_ls=("CARDINAL",)):
        block_set = set(block_ls)
        tagged_dict2 = {}
        for sent in tagged_dict:
            (slot_values, slot_types) = tagged_dict[sent]
            slot_values2, slot_types2 = [], []

            # block certain tagged slots
            for i, slot_type in enumerate(slot_types):
                if slot_type not in block_set:
                    slot_values2 += [slot_values[i]]
                    slot_types2 += [slot_types[i]]
            # update map
            tagged_dict2[sent] = (tuple(slot_values2), tuple(slot_types2))

        return tagged_dict2


if __name__ == '__main__':
    sentences = [
        "i will not need the car i rented in vienna anymore",
        "fraud claim under two zero three four eight five three four one",
        "request payment for group dinner from the johnson family",
    ]

    ner_tagger = POSTagger(tagger_id="flair/pos-english")
    tagged_dict = ner_tagger.tag(sentences)

    print(tagged_dict)
    tagged_dict2 = ner_tagger.batch_tag(sentences, start=0, end=3)
    print(tagged_dict2)
    tagged_dict3 = ner_tagger.batch_find_keysent(sentences, start=0, end=3)
    print(tagged_dict3)

