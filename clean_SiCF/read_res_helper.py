import json
import pandas as pd
import os

def read_force_eval_res(file_path, file_name, xlsx_save_name):
    input_file = os.path.join(file_path, file_name)
    target_name = xlsx_save_name
    with open(input_file, 'r') as f:
        content = f.read()
    metric_scores = json.loads(content)
    keywords = metric_scores[0].keys()
    keywords = list(keywords)
    res_dict = {}
    for ele in keywords:
        res_dict[ele] = []
    for metric_score in metric_scores:
        for ele in keywords:
            cur_res = metric_score[ele]
            res_dict[ele].append(cur_res)
    target_file = os.path.join(file_path, target_name)
    writer = pd.ExcelWriter(target_file)

    res_dict_csv = pd.DataFrame(res_dict)
    res_dict_csv.to_excel(writer, sheet_name='main')

    # writer.save()
    writer.close()
    print(f"result csv is saved to {target_file}")

if __name__ == "__main__":
    root_path = 'XXXX'
    file_path = 'Models/newSAMSUM2_DialogLED_base_SSDS_max_800_maxlen_96_min5_merge_cluster_ori'
    root_file_path = os.path.join(root_path, file_path)
    file_name = 'sicf_res.json'
    read_force_eval_res(root_file_path, file_name)




