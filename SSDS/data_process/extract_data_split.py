


import json
import os

def list_to_json(p_list, file_name):
    if os.path.isfile(file_name):
        os.remove(file_name)
    with open(file_name, 'a') as f:
        for ele in p_list:

            json_str = json.dumps(ele)
            f.write(json_str)
            f.write('\n')
    print(f"{file_name} has been generated")

def json_to_list(file_name):
    all_tr_list = []
    try:

        with open(file_name, 'r') as ini_f:
            ini_tr_files = ini_f.readlines()
            for tr_file in ini_tr_files:
                cur_tr = json.loads(tr_file)
                all_tr_list.append(cur_tr)
    except:

        all_tr_list = json.loads(file_name)
    return all_tr_list

def transfer_raw_format(root_path, ori_file, tar_path):
    ori_path = os.path.join(root_path, ori_file)
    save_path = os.path.join(tar_path, ori_file)
    with open(ori_path, 'r') as f:
        ori_data = json.load(f)

    ori_list = []
    for ele in ori_data:
        ori_list.append(ele)
    list_to_json(p_list=ori_list, file_name=save_path)
    print(f"{ori_path} has been saved to {save_path}")
    return ori_list


def main():
    root_path = '../raw_data/'
    ori_train_file = 'train.json'
    ori_val_file = 'val.json'
    ori_test_file = 'test.json'

    tar_path = './proSAMSUM'


    # step1: transfer to 1 row format
    ori_train_list = transfer_raw_format(root_path, ori_train_file, tar_path)
    ori_val_list = transfer_raw_format(root_path, ori_val_file, tar_path)
    ori_test_list = transfer_raw_format(root_path, ori_test_file, tar_path)

    # step2: divide unlabeled data (latter 50%)
    train_size = len(ori_train_list)
    unlabel_train_data = ori_train_list[int(train_size*0.5):]
    unlabel_train_data_file = 'gt_unlabeled_train.json'
    unlabel_train_data_path = os.path.join(tar_path, unlabel_train_data_file)
    list_to_json(p_list=unlabel_train_data, file_name=unlabel_train_data_path)

    # step3: divide labeled data (first 1%, 5%)
    label_train_data_file1 = 'labeled_train_1per.json'
    label_train_data_file5 = 'labeled_train_5per.json'

    label_train_data1 = ori_train_list[0:int(train_size*0.01)]
    label_train_data_file1 = 'labeled_train_1per.json'
    label_train_data_path1 = os.path.join(tar_path, label_train_data_file1)
    list_to_json(p_list=label_train_data1, file_name=label_train_data_path1)

    label_train_data5 = ori_train_list[0:int(train_size*0.05)]
    label_train_data_file5 = 'labeled_train_5per.json'
    label_train_data_path5 = os.path.join(tar_path, label_train_data_file5)
    list_to_json(p_list=label_train_data5, file_name=label_train_data_path5)


if __name__ == "__main__":
    main()