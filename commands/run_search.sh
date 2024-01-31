
echo "task=$task"
echo "template_idx=$template_idx"
echo "method=$method"
echo "progressive_p=$progressive_p"
echo "initial_indication_set_size=$initial_indication_set_size"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "ptm_name=$ptm_name"
echo "final_candidate_size=$final_candidate_size"
echo "label_balance=$label_balance"
echo "candidate_example_num_total=$candidate_example_num_total"
echo "candidate_example_num_every_label=$candidate_example_num_every_label"
echo "direct_plus=$direct_plus"
echo "diversity_score_scale=$diversity_score_scale"


ptm_name_2=$(echo $ptm_name | sed 's/\//_/g')
# if [ $ptm_name = "EleutherAI/gpt-neo-1.3B" ]
# then
#   ptm_name_2="EleutherAI_gpt-neo-1.3B"
#   else

#     ptm_name_2=$ptm_name
# fi

#for method in direct channel
search_seed=100
indication_order_random_seed=100
progressive_exp_tag=2022_1210_final
progressive_select_metric=loss

input_dir="exps/tag_$progressive_exp_tag/${ptm_name_2}/${task}/$method/tempalte_${template_idx}/seed_${indication_order_random_seed}_use_${progressive_select_metric}_p_${progressive_p}.0_initial_i_size_${initial_indication_set_size}_final_c_size_${final_candidate_size}_balance_1"
mkdir -p $input_dir
echo "input_dir=$input_dir"


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main_for_iterative_optimize_train_acc_balance.py \
    --direct_plus $direct_plus \
    --num_iteration 10 \
    --input_dir $input_dir \
    --seed $search_seed \
    --task $task \
    --method $method \
    --template_idx $template_idx \
    --which_normalization std \
    --diversity_score_scale $diversity_score_scale \
    --label_balance $label_balance \
    --candidate_example_num_total $candidate_example_num_total \
    --candidate_example_num_every_label $candidate_example_num_every_label \
    --which_diversity_feature lm_feedback \
    --beam_size 8 \
    --select_metric loss \
    --sample_topk_range 4 \
    --num_indication 100 \
    --train_data_shuffle_and_select_indication_data_seed 100 \
    --verify_metric acc \
    --ptm_name $ptm_name \
    --batch_size 36 \
    --shuffle_order_num_in_beam 2 \
    --initial_example_idxs searched_from_heuristic_score \
    --initial_example_idxs_repeat_limit 4