#!/bin/bash
#
dname=$(dirname "$PWD")
cd $dname
echo $dname

# general related
ft_ori_model=true
ft_semi_model=true
datasetname="newDIALOGSUM"
use_merge=true # if false then use seperate mode
train_percent=1 # [1, 5] only appplicalbe to datasetname="proSAMSUM"
#labeled_train_ratio=0.1 # because use preset datasplit, the labeled_train_ratio is usefulness

use_sample_selection=true
select_ratio_on_sicf=25

val_file="data/${datasetname}/val.json"
test_file="data/${datasetname}/test.json"

if ${use_merge}
then
  if [ ${train_percent} == 1 ];
  then
    general_output_dir="Models/${datasetname}_DialogLED_base_apply_sicfscore_merge_cluster"
  elif [ ${train_percent} == 5 ];
  then
    general_output_dir="Models/${datasetname}_DialogLED_base_apply_sicfscore_merge_5per_cluster"
  fi
else
general_output_dir="Models/${datasetname}_DialogLED_base_SSDS_max_800_maxlen_96_sep"
fi

# ori related
ori_ini_model_name_or_path="MingZhong/DialogLED-base-16384" # initital model as
ori_output_dir="${general_output_dir}_ori" # where well-trained model is saved
ori_ini_tr_file="data/${datasetname}/train.json" # ori train data provided by the dataset

if [ ${train_percent} == 1 ];
then
  ori_gt_labtr_file="data/${datasetname}/labeled_train_1per.json" # divided the original: 1% of original used as labeled data
elif [ ${train_percent} == 5 ];
then
  ori_gt_labtr_file="data/${datasetname}/labeled_train_5per.json" # divided the original: 5% of original used as labeled data
fi

ori_gt_unltr_file="data/${datasetname}/gt_unlabeled_train.json" # divided the original: other one part used as unlabeled data with its ground-truth labels
ori_psu_unltr_file="pseudo_unlabeled_train.json" # divided the original: other one part used as unlabeled data with its pseudo labels



# semi related

# used for sicf (need revise)
unc_save_prefix="newDIALOGSUM_1p10t_full_bnn_mean_dp_select_linearrank_aws" # using new cov by POS


if ${use_sample_selection}
then
semi_output_dir="${general_output_dir}_semi_bertscore_wi_sicf_${unc_save_prefix}_${select_ratio_on_sicf}" # where well-trained model is saved
else
semi_output_dir="${general_output_dir}_semi_bertscore_wi_sicf_${unc_save_prefix}" # where well-trained model is saved
fi

merge_psu_unltr_file_prefix="merge_pseudo_unlabeled_train"
merge_psu_unltr_file="${merge_psu_unltr_file_prefix}.json" # the file to save the sicf

# merge related
semi_ini_model_name_or_path=${ori_ini_model_name_or_path} # initital model as
if ${use_sample_selection}
then
# for selection
semi_psu_unltr_file="${ori_output_dir}/${merge_psu_unltr_file_prefix}_wi_sampleweight_${unc_save_prefix}_select_${select_ratio_on_sicf}.json" # merge case # used for sicf_weight case
else
# for pure weigh
semi_psu_unltr_file="${ori_output_dir}/${merge_psu_unltr_file_prefix}_wi_sampleweight_${unc_save_prefix}.json" # merge case # used for sicf_weight case
fi


export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0,1,2,3 \



if ${ft_ori_model}
then
echo "start fine tune ori model"

python SSDS_main.py \
    --model_name_or_path Models/newDIALOGSUM_DialogLED_base_apply_sicfscore_merge_cluster_ori/checkpoint-800 \
    --do_train False \
    --do_eval False \
    --do_predict False \
    --do_gen_pseudo_label False \
    --do_measure_pseudo_label True \
    --train_file ${ori_gt_labtr_file} \
    --validation_file ${val_file} \
    --test_file ${test_file} \
    --text_column src \
    --summary_column tgt \
    --output_dir ${ori_output_dir} \
    --label_smoothing_factor 0.1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_source_length 1024 \
    --max_target_length 128 \
    --learning_rate 2e-5 \
    --warmup_steps 50 \
    --max_steps 800 \
    --save_strategy steps \
    --evaluation_strategy steps \
    --save_steps 50 \
    --eval_steps 50 \
    --fp16 \
    --predict_with_generate \
    --round_name labeled \
    --div_train False \
    --ini_train_file ${ori_ini_tr_file} \
    --gt_unlabeled_train_file ${ori_gt_unltr_file} \
    --pseudo_unlabeled_train_file ${ori_psu_unltr_file} \
    --merge_pseudo_unlabeled_train_file ${merge_psu_unltr_file} \
    --load_best_model_at_end True \
    --metric_for_best_model BERTScore_F \
    --search_sample_mode beam_sample \
    --search_sampling_num 4 \
    --num_beams 6 \
    --search_sampling_filter_thrs 0.3 \
    --length_penalty 2.0 \
    --search_sample_mode beam_sample \
    --search_sampling_temp 0.7 \
    --use_small False \
    --sicf_pse_unlabeled_train_file sicf_pse_unlabeled_train.json \
    --sicf_merge_pseudo_unlabeled_train_file sicf_merge_pseudo_unlabeled_train.json \
    --sampling_case 1p10t \
    --dataname newDIALOGSUM \
    --cal_sein_emb False \
    --cal_cov_emb False \
    --cal_fai_emb False \
    --sicf_mode sein_cov_fai \
    --unc_type bnn_mean_dp \
    --use_sicf_rank True \
    --unc_save_prefix ${unc_save_prefix} \
    --sein_coeff 0.25 \
    --cov_coeff 0.5 \
    --fai_coeff 0.25 \
    --select_ratio_on_sicf ${select_ratio_on_sicf} \





echo "end fine tune ori model"
fi

if ${ft_semi_model}
then
echo "*****************************************"
echo "start fine tune semi model"
#python SSDS_main_mode_choose.py \
python SSDS_main.py \
    --model_name_or_path ${semi_ini_model_name_or_path} \
    --do_train True \
    --do_eval True \
    --do_predict True \
    --train_file ${semi_psu_unltr_file} \
    --validation_file ${val_file} \
    --test_file ${test_file} \
    --text_column src \
    --summary_column tgt \
    --output_dir ${semi_output_dir} \
    --label_smoothing_factor 0.1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --max_source_length 1024 \
    --max_target_length 128 \
    --learning_rate 2e-5 \
    --warmup_steps 50 \
    --max_steps 8000 \
    --save_strategy steps \
    --evaluation_strategy steps \
    --save_steps 250 \
    --eval_steps 250 \
    --fp16 \
    --predict_with_generate \
    --round_name unlabeled \
    --div_train False \
    --load_best_model_at_end True \
    --metric_for_best_model BERTScore_F \
    --use_sicf_ssds True \

echo "end fine tune semi model"
fi

echo "ori_output_dir is ${ori_output_dir}"
echo "semi_output_dir is ${semi_output_dir}"

