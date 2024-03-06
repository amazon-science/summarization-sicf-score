# Semi-Supervised Dialogue Abstractive Summarization via Selecting High-Quality Pseudolabels using SiCF Scores
This is Jianfeng He's intern project related to Semi-Supervised Dialogue Abstractive Summarization via Selecting High-Quality 
Pseudolabels using SiCF Scores. 

## Abastract
Semi-supervised dialogue summarization (SSDS) leverages model-generated summaries to reduce reliance on human-labeled data and improve the performance of summarization models. While addressing label noise, previous works on semi-supervised learning primarily focus on natural language understanding tasks, assuming each sample has a unique label. However, these methods are not directly applicable to SSDS, as it is a generative task, and each dialogue can be summarized in different ways. In this work, we propose a novel scoring approach, SiCF, which encapsulates three primary dimensions of summarization model quality: Semantic invariance (indicative of model confidence), Coverage (factual recall), and Faithfulness (factual precision). Using the SiCF score, we select unlabeled dialogues with high-quality generated summaries to train summarization models. Comprehensive experiments on three public datasets demonstrate the effectiveness of SiCF scores in uncertainty estimation and semi-supervised learning for dialogue summarization tasks.

## Env
```bash
## if not yet install
conda env create -f dialogue0_v2.yaml

conda activate dialogue0

```

## Data 
The original data is saved as below.
Original data with similar data process as CODA
```bash
./data/newSAMSUM2/
./data/newDIALOGSUM/
./data/newTODSUM/
```

Beam search sampling data is saved in 
```bash
# 1:50, 5:50
./data/sampled_data/newSAMSUM2/ 
# 1:50, 5:50
./data/sampled_data/newDIALOGSUM/
# 2:90, 10:90
./data/sampled_data/newTODSUM/
```

## Train initial dialogue summarization model
```shell
cd ./script

## SAMSUM 1percent 25% selection
bash aws_sicfssds_samsum2_mid_sec_bertscore_sicf_sein_cov_fai_1p10t_linearsim_samsel_5_25_full.sh

## DialogSUM 1percent 25% selection
bash aws_dialogsum_sicfssds_mid_sec_bertscore_sicf_sein_cov_fai_1p10t_linearsim_samsel_5_25_full.sh

## TODSUM 1percent 25% selection
bash aws_todsum_sicfssds_mid_sec_bertscore_sicf_sein_cov_fai_1p10t_linearsim_samsel_5_25_full.sh
```
Please replace the respective part of first py script as
```bash
    --do_train True \
    --do_eval True \
    --do_predict True \
    --do_gen_pseudo_label False \
    --do_measure_pseudo_label False \
```

## Beam search sampling
```shell
cd ./script

## SAMSUM 1percent 25% selection
bash aws_sicfssds_samsum2_mid_sec_bertscore_sicf_sein_cov_fai_1p10t_linearsim_samsel_5_25_full.sh

## DialogSUM 1percent 25% selection
bash aws_dialogsum_sicfssds_mid_sec_bertscore_sicf_sein_cov_fai_1p10t_linearsim_samsel_5_25_full.sh

## TODSUM 1percent 25% selection
bash aws_todsum_sicfssds_mid_sec_bertscore_sicf_sein_cov_fai_1p10t_linearsim_samsel_5_25_full.sh
```
Please replace the respective part of first py script as
```bash
    --do_train False \
    --do_eval False \
    --do_predict False \
    --do_gen_pseudo_label True \
    --do_measure_pseudo_label False \
```



## Two tasks
Our work relate to two tasks: uncertainty estimation on dialogue summarization and semi-supervised dialogue summarization (SSDS).
And these two tasks can be together run by one script as below.
```shell
cd ./script

## SAMSUM 1percent 25% selection
bash aws_sicfssds_samsum2_mid_sec_bertscore_sicf_sein_cov_fai_1p10t_linearsim_samsel_5_25_full.sh

## DialogSUM 1percent 25% selection
bash aws_dialogsum_sicfssds_mid_sec_bertscore_sicf_sein_cov_fai_1p10t_linearsim_samsel_5_25_full.sh

## TODSUM 1percent 25% selection
bash aws_todsum_sicfssds_mid_sec_bertscore_sicf_sein_cov_fai_1p10t_linearsim_samsel_5_25_full.sh
```
The results are saved in a folder strongly related to **unc_save_prefix** in the above scripts.

We list the examples as below.

For example, you can read the results of uncertainty estimation on dialogue summarization in:
```shell
cd ./Models

## SAMSUM 1percent 25% selection
basic_folder="newSAMSUM2_DialogLED_base_apply_sicfscore_merge_cluster_ori"

## DialogSUM 1percent 25% selection
basic_folder="newDIALOGSUM_DialogLED_base_apply_sicfscore_merge_cluster_ori"

## TODSUM 1percent 25% selection
basic_folder="newTODSUM_DialogLED_base_apply_sicfscore_merge_cluster_ori"

## read uncertainty estimation results results
# saved in "${basic_folder}"/"${unc_save_prefix}"_force_eval_res.xlsx"
# please download it and read
```

For example, you can read the results of SSDS in below path:
```shell
cd ./Models

## SAMSUM 1percent 25% selection
cd ./newSAMSUM2_DialogLED_base_apply_sicfscore_merge_cluster_semi_bertscore_wi_sicf_newSAMSUM2_1p10t_full_bnn_mean_dp_select_linearrank_aws_25

## DialogSUM 1percent 25% selection
cd ./newDIALOGSUM_DialogLED_base_apply_sicfscore_merge_cluster_semi_bertscore_wi_sicf_newDIALOGSUM_1p10t_full_bnn_mean_dp_select_linearrank_aws_25

## TODSUM 1percent 25% selection
cd ./newTODSUM_DialogLED_base_apply_sicfscore_merge_cluster_semi_bertscore_wi_sicf_newTODSUM_1p10t_full_bnn_mean_up_select_linearrank_aws_25

## read SSDS results
vim all_results.json
```
Once you have opened the **all_results.json**, where please refer to **unlrd_** related results, which are the results of the semi-supervised training results
