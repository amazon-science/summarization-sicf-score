from dataclasses import dataclass

from flair.data import Sentence
from flair.models import SequenceTagger

import nltk

@dataclass
class NERTagger:
    """NER tagger based on flair package"""

    tagger_id: str = "flair/ner-english-ontonotes" 

    def __post_init__(self):
        self.tagger = SequenceTagger.load(self.tagger_id)

    def tag(self, sentences):
        # output format: a map of sentence -> (slot_values, slot_types)
        sentences = [Sentence(text) for text in sentences]
        self.tagger.predict(sentences)

        tagged_dict = {}
        for sent in sentences:
            spans = sent.get_spans("ner")
            slot_values = tuple(span.text for span in spans)
            slot_types = tuple(span.tag for span in spans)
            tagged_dict[sent.text] = (slot_values, slot_types)

        return tagged_dict

    def batch_tag(self, sentences, start, end):
        # output format: a map of sentence -> (slot_values, slot_types)
        sentences = [Sentence(text) for text in sentences]
        self.tagger.predict(sentences)

        tagged_dict = {}
        sent_len = len(sentences)
        index = list(range(start, end))
        for i in range(len(sentences)):
            sent = sentences[i]
            spans = sent.get_spans("ner")
            slot_values = tuple(span.text for span in spans)
            slot_types = tuple(span.tag for span in spans)
            tagged_dict[index[i]] = (slot_values, slot_types)

        return tagged_dict

    def batch_find_keysent(self, sentences, start, end):
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
                spans = subsent.get_spans("ner")
                res[subsent.text] = len(spans) # the span length is the importance of the subsent.
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

    ner_tagger = NERTagger(tagger_id="ner-ontonotes")
    tagged_dict = ner_tagger.tag(sentences)
    processed_tagged_dict = ner_tagger.block_slot_types(tagged_dict)
    print(tagged_dict)
