import os
import numpy as np
import torch, gc
from transformers.trainer_utils import EvalPrediction

from transformers import set_seed
import random

def train(training_args, data_args, last_checkpoint, trainer, train_dataset):
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    best_model_save_path = os.path.join(training_args.output_dir, 'ck_best')
    trainer.save_model(output_dir=best_model_save_path)  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    return trainer

def val(data_args, model_args, trainer, eval_dataset, logger):
    logger.info("*** Evaluate ***")

    metrics = trainer.evaluate(
        max_length=data_args.val_max_target_length,
        num_beams=data_args.num_beams,
        metric_key_prefix="eval",
        length_penalty=model_args.length_penalty, # this length_penalty is added by me
    )
    gc.collect()
    torch.cuda.empty_cache()
    max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    return trainer, logger


def test(training_args, data_args, model_args, trainer, predict_dataset, tokenizer, logger, metric_key_prefix, test_gen_pred_file_name):
    logger.info("*** Predict ***")

    ### original
    predict_results = trainer.predict(
        predict_dataset,
        # metric_key_prefix="predict", # original
        metric_key_prefix=metric_key_prefix,
        max_length=data_args.val_max_target_length,
        num_beams=data_args.num_beams, # original: None
        length_penalty=model_args.length_penalty,
    )
    gc.collect()
    torch.cuda.empty_cache()

    

    metrics = predict_results.metrics
    max_predict_samples = (
        data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
    )
    metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

    # trainer.log_metrics("predict", metrics) # original
    # trainer.save_metrics("predict", metrics) # original
    trainer.log_metrics(metric_key_prefix, metrics)
    trainer.save_metrics(metric_key_prefix, metrics)

    if trainer.is_world_process_zero():
        if training_args.predict_with_generate:

            try:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
            except:
                print('************there are test decode error************')
                print(predict_results.predictions.shape)
                np.savetxt('./test_decode_error.txt', predict_results.predictions)
                print('save wrong info to ./decode_error.txt')
                # https://github.com/huggingface/transformers/issues/22634
                preds = np.where(predict_results.predictions != -100, predict_results.predictions, tokenizer.pad_token_id)
                predictions = tokenizer.batch_decode(preds, skip_special_tokens=True)

                print(f"preds is {preds}")




            predictions = [pred.strip() for pred in predictions]
            output_prediction_file = os.path.join(training_args.output_dir, test_gen_pred_file_name)
            with open(output_prediction_file, "w") as writer:
                writer.write("\n".join(predictions))

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)
    return trainer, logger

def gen_psudo(training_args, data_args, model_args, trainer, predict_dataset, tokenizer, logger, metric_key_prefix, test_gen_pred_file_name, dec_samp_helper):
    logger.info("*** Predict ***")

    sampled_predres_prediction_list=[]
    sampled_predres_label_list=[]

    if model_args.use_parallel_sampling == True:
        assert model_args.parallel_sampling_id is not None
        set_seed(model_args.parallel_sampling_id)
        random.seed(model_args.parallel_sampling_id)

    for _ in range(model_args.search_sampling_num):
        print(f'cur is {_} sampling round')
        if model_args.search_sample_mode == None:
            predict_results = trainer.predict(
                predict_dataset,

                metric_key_prefix=metric_key_prefix,
                max_length=data_args.val_max_target_length,
                num_beams=data_args.num_beams, # original: None
                length_penalty=model_args.length_penalty,
            )
            sampled_predres_prediction_list.append(predict_results.predictions)
            sampled_predres_label_list.append(predict_results.label_ids)
            print(predict_results.predictions.shape)
            break 

        elif model_args.search_sample_mode == 'beam_sample':
            predict_results = trainer.predict(
           
                predict_dataset,
                
                metric_key_prefix=metric_key_prefix,
                max_length=data_args.val_max_target_length,
                num_beams=data_args.num_beams, 

                do_sample=True, 
                temperature=model_args.search_sampling_temp,
                
                length_penalty=model_args.length_penalty,
                
                num_return_sequences=1,
            )
            sampled_predres_prediction_list.append(predict_results.predictions)
            sampled_predres_label_list.append(predict_results.label_ids)
            print(predict_results.predictions.shape)

        elif model_args.search_sample_mode == 'top_kp_sample':

            predict_results = trainer.predict(
                predict_dataset,

                metric_key_prefix=metric_key_prefix,
                max_length=data_args.val_max_target_length,
                num_beams=1, 
                top_k=40,
                top_p=0.8,
                do_sample=True,
                temperature=model_args.search_sampling_temp,
                num_return_sequences=1, 
                length_penalty=model_args.length_penalty,

            )
            sampled_predres_prediction_list.append(predict_results.predictions)
            sampled_predres_label_list.append(predict_results.label_ids)
            print(predict_results.predictions.shape)

        elif model_args.search_sample_mode == 'diverse_beam':

            predict_results = trainer.predict(

                predict_dataset,

                metric_key_prefix=metric_key_prefix,
                max_length=data_args.val_max_target_length,
                num_beams=data_args.num_beams, 

                do_sample=False, 
                temperature=model_args.search_sampling_temp,
                
                length_penalty=model_args.length_penalty,
                
                num_return_sequences=1,
                num_beam_groups=model_args.diverse_beam_groups,
            )
            sampled_predres_prediction_list.append(predict_results.predictions)
            sampled_predres_label_list.append(predict_results.label_ids)
            print(predict_results.predictions.shape)

        else:
            raise ValueError(f"model_args.search_sample_mode should be in range of [None, beam_sample]")
        gc.collect()
        torch.cuda.empty_cache()

    # start choose which is the best

    sampling_txt_list_save_path = os.path.join(training_args.output_dir, model_args.search_sampling_text_save_name)
    best_sampled_txtid, best_sampled_pred, best_sampled_label, best_score_1d, pred_score_mat \
        = dec_samp_helper.score_decoding_sampling(
        pred_ids_list=sampled_predres_prediction_list,
        label_ids_list=sampled_predres_label_list,
        pred_txt_list_save_path=sampling_txt_list_save_path,
        nlg_metric=model_args.nlg_metric,
    )


    filtered_sampled_txtid, filtered_sampled_pred, filtered_sampled_label, filtered_sampled_id \
        = dec_samp_helper.filter_weigh_on_scores(
        input_txtids=best_sampled_txtid,
        input_preds=best_sampled_pred,
        input_labels=best_sampled_label,
        input_scores=best_score_1d,
        filter_mode=model_args.search_sampling_filter_mode,
        filter_thrs=model_args.search_sampling_filter_thrs
    )


    filtered_metrics = dec_samp_helper.compute_metrics(pred_ids=filtered_sampled_txtid, decoded_preds=filtered_sampled_pred, decoded_labels=filtered_sampled_label)

    max_filtered_predict_samples = len(filtered_sampled_pred)
    filtered_metrics["filtered_predict_samples"] = max_filtered_predict_samples
    print(f"max_filtered_predict_samples={max_filtered_predict_samples}")
    trainer.log_metrics(metric_key_prefix, filtered_metrics)
    trainer.save_metrics(metric_key_prefix, filtered_metrics)



    if trainer.is_world_process_zero():
        if training_args.predict_with_generate:
            output_prediction_file = os.path.join(training_args.output_dir, test_gen_pred_file_name)
            with open(output_prediction_file, "w") as writer:
                filtered_sampled_pred_write = [filtered_sampled_text.replace('\n', ' ') for filtered_sampled_text in filtered_sampled_pred]
                writer.write("\n".join(filtered_sampled_pred_write))

    
    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)



    return trainer, logger, filtered_sampled_id