# Semi-Supervised Dialogue Abstractive Summarization via Selecting High-Quality Pseudolabels using SiCF Scores
This is Jianfeng He's intern project related to Semi-Supervised Dialogue Abstractive Summarization via Selecting High-Quality 
Pseudolabels using SiCF Scores. 

## Abastract
Semi-supervised dialogue summarization (SSDS) leverages model-generated summaries to reduce reliance on human-labeled data and improve the performance of summarization models. While addressing label noise, previous works on semi-supervised learning primarily focus on natural language understanding tasks, assuming each sample has a unique label. However, these methods are not directly applicable to SSDS, as it is a generative task, and each dialogue can be summarized in different ways. In this work, we propose a novel scoring approach, SiCF, which encapsulates three primary dimensions of summarization model quality: Semantic invariance (indicative of model confidence), Coverage (factual recall), and Faithfulness (factual precision). Using the SiCF score, we select unlabeled dialogues with high-quality generated summaries to train summarization models. Comprehensive experiments on three public datasets demonstrate the effectiveness of SiCF scores in uncertainty estimation and semi-supervised learning for dialogue summarization tasks.

## Env
```bash
conda env create -f dialogue0_v2.yaml

conda activate dialogue0

```

## If you are only insterested in SiCF score for text/dialogue summary evaluation
Then, only **./sicf_example.py/** and **./clean_SiCF/** are related.
And you can get the usage of our SiCF score via running 
```bash
python sicf_example.py
```
The parameter introductions are detailed in the **./sicf_example.py/** as well.

## If you are only insterested in SiCF score for semi-supervised dialogue summarization (SSDS)
Then, please move to **./SSDS/** folder for a more detailed README introduction. 