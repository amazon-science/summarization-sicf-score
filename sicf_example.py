from clean_SiCF import sicf_score, unc_metrics
import numpy as np
import argparse
import os

def SiCFScore(sampling_summary_list, dialogue_list, args):
    ## We have provided respective examples in this .py file
    # sampling_summary_list: a list of generated summaries, where each element is a one-round result.
    #                        The sampling_summary_list is applicable to both with/without sampling generated summaries.
    # dialogue_list: a list of original text/dialouges (our SiCFScore can evaluate the generated summaries without referring to the ground-truth summaries)
    # args: related running parameters

    sicf_tool = sicf_score.SiCF_Tool(
        save_path=args.output_dir,
        use_cov_count=args.use_cov_count,
        use_fai_count=args.use_fai_count,
        use_small=args.use_small,
        prefix=args.sampling_case + '_' + args.dataname,
        fai_cal_type=args.fai_cal_type,
        sein_dis='euclidean',
        cov_dis='euclidean',
        fai_dis='euclidean',
    )

    sein_wei = 0
    sein_save_path = os.path.join(args.output_dir, 'sein_stat.npz')
    if 'sein' in args.sicf_mode:
        # cal semantic inv
        sem_inv_mic, sem_inv_pool, sem_dist_arr, sem_disrank_list = sicf_tool.txt_semantic_invariance(txt_list=sampling_summary_list,
                                                                                                      cal_sein_emb=args.cal_sein_emb,
                                                                                                      model_name=args.sein_model_name)
        # sem_inv_mic: b * d arraies, where each d-dim array shows semantic invarance for a sample in a micro-level
        # sem_inv_pool: b-length list of scalars, each scalar shows semantic invarance for a sample
        # sem_dist_arr: b * m arraies, where each row shows the distance from m samples to the
        # sem_disrank_list: b-length list of scalars, each scalar shows semantic invarance for a sample

        # save score
        np.savez(sein_save_path,
                 sem_inv_mic=sem_inv_mic,
                 sem_inv_pool=sem_inv_pool,
                 sem_dist_arr=sem_dist_arr,
                 sem_disrank_list=sem_disrank_list
                 )

        # get weights
        sein_wei = sem_dist_arr[list(range(sem_dist_arr.shape[0])), sem_disrank_list]
        sein_wei = np.array(sein_wei)

        print(f"sein_wei is {sein_wei}")
        print('finished semantic invariance scoring!')

    ## coverage
    cov_wei = 0
    if 'cov' in args.sicf_mode:

        # cal a list (#round) of cov score list (#sample)
        all_cov_score_list = sicf_tool.txt_coverage(src_list=sampling_summary_list,
                               ref=dialogue_list,
                               cal_cov_emb=args.cal_cov_emb,
                               batch_size=args.cov_batch_size,
                               )

        if sicf_tool.cov_dis in ['euclidean']:
            use_reciprocal=True
        elif sicf_tool.cov_dis in ['cosine']:
            use_reciprocal=True #

        # calculate the uncertainty scores from view of coverage
        if args.unc_type == 'bnn_dropout':
            # (to do) figure out whether using softmax and whether multi-label uncertainty
            unc_cov_score_list = unc_metrics.multilabel_bnn_dp_cal_unc_score_list(all_cov_score_list, use_reciprocal=use_reciprocal, mode=args.multi_cal_type)

            # read the uncertainty scores
            cov_wei = unc_metrics.multilabel_read_uncer_score_list(unc_cov_score_list, index_enc=args.bnn_type)
        elif args.unc_type == 'bnn_var':
            cov_wei = unc_metrics.multilabel_bnn_var_cal(all_cov_score_list, mode=args.multi_cal_type)
        elif args.unc_type == 'bnn_mean':
            unc_cov_score_list, cov_wei = unc_metrics.bnn_mean_cal_unc_score_list(all_cov_score_list)
        elif args.unc_type == 'bnn_mean_dp':
            unc_cov_score_list_mean, cov_wei_mean = unc_metrics.bnn_mean_cal_unc_score_list(all_cov_score_list)
            unc_cov_score_list_dp = unc_metrics.multilabel_bnn_dp_cal_unc_score_list(all_cov_score_list, use_reciprocal=use_reciprocal, mode=args.multi_cal_type)
            cov_wei_dp = unc_metrics.multilabel_read_uncer_score_list(unc_cov_score_list_dp, index_enc=args.bnn_type)
            cov_wei = list(np.array(cov_wei_mean) * np.array(cov_wei_dp))
        elif args.unc_type == 'bnn_mean_var':
            unc_cov_score_list_mean, cov_wei_mean = unc_metrics.bnn_mean_cal_unc_score_list(all_cov_score_list)
            cov_wei_var = unc_metrics.multilabel_bnn_var_cal(all_cov_score_list, mode=args.multi_cal_type)
            cov_wei = list(np.array(cov_wei_mean) * np.array(cov_wei_var))

        cov_wei = np.array(cov_wei)
        if args.use_small:
            print(f"cov_wei is {cov_wei}")

    ## faithfulness
    fai_wei = 0
    if 'fai' in args.sicf_mode:
        all_fai_score_list = sicf_tool.txt_faithfulness(src_list=sampling_summary_list,
                               ref=dialogue_list,
                               cal_fai_emb=args.cal_fai_emb,
                               batch_size=args.fai_batch_size,
                               )

        if sicf_tool.fai_dis in ['euclidean']:
            use_reciprocal=True
        elif sicf_tool.fai_dis in ['cosine']:
            use_reciprocal=True

        # calculate the uncertainty scores from view of faithfulness
        if args.unc_type=='bnn_dropout':
            unc_fai_score_list = unc_metrics.multilabel_bnn_dp_cal_unc_score_list(all_fai_score_list, use_reciprocal=use_reciprocal, mode=args.multi_cal_type)
            fai_wei = unc_metrics.multilabel_read_uncer_score_list(unc_fai_score_list, index_enc=args.bnn_type)
        elif args.unc_type == 'bnn_var':
            fai_wei = unc_metrics.multilabel_bnn_var_cal(all_fai_score_list, mode=args.multi_cal_type)
        elif args.unc_type == 'bnn_mean':
            unc_fai_score_list, fai_wei = unc_metrics.bnn_mean_cal_unc_score_list(all_fai_score_list)
        elif args.unc_type == 'bnn_mean_dp':
            unc_fai_score_list_mean, fai_wei_mean = unc_metrics.bnn_mean_cal_unc_score_list(all_fai_score_list)
            unc_fai_score_list_dp = unc_metrics.multilabel_bnn_dp_cal_unc_score_list(all_fai_score_list, use_reciprocal=use_reciprocal, mode=args.multi_cal_type)
            fai_wei_dp = unc_metrics.multilabel_read_uncer_score_list(unc_fai_score_list_dp, index_enc=args.bnn_type)
            fai_wei = list(np.array(fai_wei_mean) * np.array(fai_wei_dp))
        else:
            raise ValueError

        fai_wei = np.array(fai_wei)
        if args.use_small:
            print(f"fai_wei is {fai_wei}")
        print('finished faithfulness scoring!')

    # merge sein, cov and fai into SiCF scores
    if args.use_sicf_rank == True:
        # normalization based on the ranks
        sein_rank = sicf_tool.wei2rank(sein_wei)
        cov_rank = sicf_tool.wei2rank(cov_wei)
        fai_rank = sicf_tool.wei2rank(fai_wei)
        total_rank = args.sein_coeff * sicf_tool.get_total_rank(sein_rank) + args.cov_coeff * sicf_tool.get_total_rank(cov_rank) + args.fai_coeff * sicf_tool.get_total_rank(fai_rank)
        SiCF_score = (args.sein_coeff * sein_rank + args.cov_coeff * cov_rank + args.fai_coeff * fai_rank) / (total_rank + 1e-8)
    else:
        # normalization based on the values
        normalize_sein = sicf_tool.normalize_wei(sein_wei)
        normalize_cov = sicf_tool.normalize_wei(cov_wei)
        normalize_fai = sicf_tool.normalize_wei(fai_wei)
        total_coef = sicf_tool.get_total_coef(sein_wei, args.sein_coeff) + sicf_tool.get_total_coef(cov_wei, args.cov_coeff) + sicf_tool.get_total_coef(fai_wei, args.fai_coeff)
        SiCF_score = (args.sein_coeff * normalize_sein + args.cov_coeff * normalize_cov + args.fai_coeff * normalize_fai) / (total_coef + 1e-8)

    return SiCF_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SiCF Example')

    ### generally related
    parser.add_argument('-output_dir', type=str, default="./sicf_example_emb", help='saving path')
    parser.add_argument('-use_small', type=bool, default=True, help='whehter use the small data to demo')
    parser.add_argument('-sampling_case', type=str, default="1p10t", help='case of percent & temparature')
    parser.add_argument('-dataname', type=str, default="newTODSUM", help='the data name')
    parser.add_argument('-seed', type=int, default=0, help='random_seed')
    parser.add_argument('-used_sampling', type=bool, default=True, help='special in this example py file and used for indication of whether used sampling generation (e.g., beam search sampling) or not.')

    ### sicf related
    parser.add_argument('-sicf_mode', type=str, default="sein_cov_fai", help='set whether use sein (semantic invariance), cov (coverage), and fai (faithfulness)')
    parser.add_argument('-use_sicf_rank', type=bool, default=True, help='True: use location rank; False: use score ')
    parser.add_argument('-multi_cal_type', type=str, default='sum', help='Set what kind of unc type to be used. [mean, sum] applied into the bnn_dropout/bnn_var')
    parser.add_argument('-bnn_type', type=int, default=0,
                        help='Set what kind of unc type to be used. [0: predictive uncertainty, 1: aleatoric uncertainty, 2: epistemic uncertainty]')
    # unc_type: {'bnn_mean_dp': 'm+BNN', 'bnn_dropout':'BNN', 'bnn_mean':'mean'}
    parser.add_argument('-unc_type', type=str, default='bnn_dropout', help='Set what kind of unc type to be used. [bnn_dropout, bnn_mean, bnn_mean_dp]')

    # semantic invariance related
    parser.add_argument('-cal_sein_emb', type=bool, default=True, help='whehter calculate the sein_emb if sein is open')
    parser.add_argument('-sein_coeff', type=float, default=0.33, help='the coeff of wei scores, smaller gives more weights')
    parser.add_argument('-sein_model_name', type=str, default="roberta-base", help='whehter measure the quality of pseudo labels')

    # coverage related
    parser.add_argument('-cal_cov_emb', type=bool, default=True, help='whehter calculate the cov_emb if cov is open')
    parser.add_argument('-cov_batch_size', type=int, default=32, help='the batch size used in the cov calculation')
    parser.add_argument('-cov_coeff', type=float, default=0.33, help='the coeff of wei scores, smaller gives more weights')
    parser.add_argument('-use_cov_count', type=bool, default=True, help='whehter use the counts of tokens to weigh the cov')

    # faithfulness related
    parser.add_argument('-cal_fai_emb', type=bool, default=True, help='whehter calculate the fai_emb if fai is open')
    parser.add_argument('-fai_batch_size', type=int, default=32, help='the batch size used in the fai calculation')
    parser.add_argument('-use_fai_count', type=bool, default=True, help='whehter use the counts of sentences to weigh the fai')
    parser.add_argument('-fai_coeff', type=float, default=0.33, help='the coeff of wei scores, smaller gives more weights')
    parser.add_argument('-fai_cal_type', type=str, default="nli", help='the way to cal fai scores [nli]')


    args = parser.parse_args()

    # generated summaries with sampling technology
    with_sampling_summary_list = [
        # round 1 sampling
        ["#Person2# tells #Person1# #Person2 #'s favorite TV program is game shows. Lucy prefers watching the international news.",
         "#Person1# phones Mr. Lewis to tell him he's going to Taipei by way of Hong Kong.",
         '#Person1# and #Person2# are planning to get to the Town Center during the day. They decide to take a taxi or the bus.',
         '#Person1# and #Person2# are going to see the movie. They decide to go hiking and have a picnic.'],
        # round 2 sampling
        ["#Person2# tells #Person1# #Person2 #'s favorite TV program but Lucy prefers watching the international news.",
         "#Person1# phones Mr. Lewis to tell him he's going to Taipei by way of Hong Kong.",
         '#Person1# and #Person2# are planning to get to the Town Center during the day. They decide to take a taxi or the bus.',
         '#Person1# and #Person2# are going to see the movie. They decide to go hiking and have a picnic.'],
        # round 3 sampling
        ["#Person2# tells #Person1# #Person2 #'s favorite TV program but Lucy prefers watching the international news.",
         "#Person1# tells Mr. Lewis he's going to Taipei by way of Hong Kong. #Person2# gives Mr.Lewis his passport and ticket.",
         '#Person1# and #Person2# are planning to go to the Town Center. They decide to take a taxi or the bus.',
         '#Person1# and #Person2# will go to see the movie, saving the planet at the rock. They can go hiking and have a picnic.']

    ]

    # generated summaries without sampling technology
    without_sampling_summary_list = [
        # without sampling and thus only one-round result
        ["#Person2# tells #Person1# #Person2 #'s favorite TV program is game shows. Lucy prefers watching the international news.",
         "#Person1# phones Mr. Lewis to tell him he's going to Taipei by way of Hong Kong.",
         '#Person1# and #Person2# are planning to get to the Town Center during the day. They decide to take a taxi or the bus.',
         '#Person1# and #Person2# are going to see the movie. They decide to go hiking and have a picnic.'],
    ]

    # original texts/dialogues
    dialogue_list = [
        "#Person1#: Hi Lucy , what's your favorite TV program? . #Person2#: I like sports programs best , especially tennis . I really prefer playing to watching . . #Person1#: What about your best friend Rosie? What does she like to watch? . #Person2#: Her favorite shows are game shows where teams answer questions and win prizes . What programs do you like best Tim? . #Person1#: Oh , I really enjoy exciting films . My best friend Carl prefers watching the international news .",
        "#Person1#: Hello . . #Person2#: Good morning , sir . Welcome to London Heathrow Airport . Where are you going? . #Person1#: I'm going to Taipei by way of Hong Kong . . #Person2#: Your passport and ticket , please . Will you be checking in any banks , Mr . Lewis? . #Person1#: Just one and I have this carry-on bag . . #Person2#: What kind of seat would you like , Mr . Lewis? . #Person1#: Window , please . . #Person2#: Your flight 923 will board at gate 35 . It is 9 o'clock now and boarding will begin in 2 hours 30 minutes before the flight takes off . And here's your boarding pass . Have a good trip , Mr . Lewis . . #Person1#: Thank you . Bye .",
        "#Person1#: Can you give me some information about getting to the Town Center? . #Person2#: Well , you can drive , but the parking is difficult . It will be quite expensive if you stay there all day . . #Person1#: Yes , we're thinking of going to look around some of the shops , and look at the wall around the city . So we'll probably be there most of the day . . #Person2#: In that case , you'd better take a taxi or the bus . . #Person1#: How much does taking a taxi cost? . #Person2#: It will be about 12 pounds . Actually I'd say it's around 16 pounds , because fares have increased recently . We can book it for you in our travel agency and it will pick you up outside . It only takes about 10 minutes . . #Person1#: Right , I see . What about taking the bus? How much is that? . #Person2#: It's only 2 pounds per person , it's not far from here . You go out of here , turn right on Oak Tree Avenue , and it's about a 5 minute walk down the road . The bus ride is about 15 minutes . . #Person1#: Oh , OK , maybe we could do that .",
        "#Person1#: We can go to see the movie , saving the planet at the rock . What time does it start? . #Person2#: 8:00 o'clock . . #Person1#: So we can be back about 10:30 , right? . #Person2#: No , it doesn't end until 11 . . #Person1#: I can't sit in the cinema so long . . #Person2#: Well then , what do you want to see? . #Person1#: Shakespeare in love is at the regal and twister at the royal . Shakespeare in love starts at 7:45 and it ends at 9:00 . . #Person2#: Ok , let's go to see Shakespeare in love . I can see saving the planet with my friend Barbara later . . #Person1#: What are we going to do after the movie? . #Person2#: We can go hiking and have a picnic ."
    ]

    if args.used_sampling:
        summary_list = with_sampling_summary_list
    else:
        summary_list = without_sampling_summary_list

    assert len(summary_list[0]) == len(dialogue_list)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # calculate SiCF score
    sicf_score_res = SiCFScore(summary_list, dialogue_list, args)
    print('\n')
    print(f"sicf_score_res for each sample is {sicf_score_res}")
