########
# This is an implementation of sample weighted trainer of huggineface
########
import torch
from torch import nn
from transformers import Trainer, Seq2SeqTrainer
from transformers.modeling_utils import  unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from packaging import version
import datasets




## below is our implementation of sample weighted seq2seq trainer
class SampleWeightedSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
                How the loss is computed by Trainer. By default, all models return the loss in the first element.

                Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # below is revised for sample weighted
        processed_sw = None
        if 'sample_weight' in inputs:
            initialized_sw = inputs.pop("sample_weight")
            processed_sw = self.process_sample_weight(initialized_sw)
            assert processed_sw is not None



        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                if processed_sw is not None:
                    # sample weighted case
                    loss = self.sample_weighted_label_smoother(outputs, labels, sample_weight=processed_sw, shift_labels=True)
                else:
                    # original case
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                if processed_sw is not None:
                    # sample weighted case
                    loss = self.sample_weighted_label_smoother(outputs, labels, sample_weight=processed_sw)
                else:
                    # original case
                    loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))

        if 'sample_weight' in ignored_columns:
            self._signature_columns.append('sample_weight')

            signature_columns = self._signature_columns

            ignored_columns = list(set(dataset.column_names) - set(signature_columns))


        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set"
            print(f"The following columns {dset_description} don't have a corresponding argument in ")
            print(f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}.")
            print(f" If {', '.join(ignored_columns)} are not expected by `{self.model.__class__.__name__}.forward`, ")
            print(" you can safely ignore this message.")


        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    def process_sample_weight(self, sample_weight):

        processed_weight = (1.0 - sample_weight) / 2 + 0.5 

        return processed_weight

    def sample_weighted_label_smoother(self, model_output, labels, sample_weight=None, shift_labels=False):

        epsilon: float = 0.1
        ignore_index: int = -100

        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(ignore_index)

        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)

        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)



        if sample_weight is not None: # check the data type of nll_loss and smoothed_loss
            nll_loss = nll_loss * sample_weight
            smoothed_loss = smoothed_loss * sample_weight


        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - epsilon) * nll_loss + epsilon * smoothed_loss



class SampleWeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
                How the loss is computed by Trainer. By default, all models return the loss in the first element.

                Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # below is revised for sample weighted
        processed_sw = None
        if 'sample_weight' in inputs:
            initialized_sw = inputs.pop("sample_weight")
            processed_sw = self.process_sample_weight(initialized_sw)


        outputs = model(**inputs)

        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                if 'sample_weight' in inputs:
                    # sample weighted case
                    loss = self.sample_weighted_label_smoother(outputs, labels, sample_weight=processed_sw,
                                                               shift_labels=True)
                else:
                    # original case
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                if 'sample_weight' in inputs:
                    # sample weighted case
                    loss = self.sample_weighted_label_smoother(outputs, labels, sample_weight=processed_sw)
                else:
                    # original case
                    loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )

            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def process_sample_weight(self, sample_weight):
        return sample_weight

    def sample_weighted_label_smoother(self, model_output, labels, sample_weight=None, shift_labels=False):
        # revised from class LabelSmoother -> __call__()


        epsilon: float = 0.1
        ignore_index: int = -100

        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(ignore_index)

        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)

        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)



        if sample_weight is not None:  
            nll_loss = nll_loss * sample_weight
            smoothed_loss = smoothed_loss * sample_weight


        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - epsilon) * nll_loss + epsilon * smoothed_loss