import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from bert_score import BERTScorer

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels



class Metrics(object):
    def __init__(self, tokenizer, data_args, metric):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.metric = metric
        self.bertscorer = BERTScorer(lang="en", rescale_with_baseline=True)

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0] # preds are array


        try:
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        except:
            print('************there are decode error************')
            print(preds.shape)
            np.savetxt('./decode_error.txt', preds)
            print('save wrong info to ./decode_error.txt')
            # https://github.com/huggingface/transformers/issues/22634
            preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

            print(f"preds is {preds}")


        if self.data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    def compute_metrics_wBERTscore(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # print(f"preds={preds}")
        if self.data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)

        bs_P, bs_R, bs_F = self.bertscorer.score(decoded_preds, decoded_labels)


        result["BERTScore_P"] = bs_P.mean().item()
        result["BERTScore_R"] = bs_R.mean().item()
        result["BERTScore_F"] = bs_F.mean().item()

        result = {k: round(v, 4) for k, v in result.items()}
        return result

