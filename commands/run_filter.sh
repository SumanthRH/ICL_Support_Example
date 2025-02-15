
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


if [ $ptm_name = "EleutherAI/gpt-neo-1.3B" ]
then
  ptm_name_2="EleutherAI_gpt-neo-1.3B"
  else
    ptm_name_2=$ptm_name
fi

#for method in direct channel
search_seed=100
indication_order_random_seed=100
progressive_exp_tag=2022_1210_final
progressive_select_metric=loss

input_dir="exps/tag_$progressive_exp_tag/${ptm_name_2}/${task}/$method/tempalte_${template_idx}/seed_${indication_order_random_seed}_use_${progressive_select_metric}_p_${progressive_p}.0_initial_i_size_${initial_indication_set_size}_final_c_size_${final_candidate_size}_balance_1"
mkdir -p $input_dir
echo "input_dir=$input_dir"



CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main_for_progressive_examples_selection.py \
--indication_order_random_seed $indication_order_random_seed \
--progressive_p $progressive_p \
--initial_indication_set_size $initial_indication_set_size \
--final_candidate_size $final_candidate_size \
--indicate_example_from random \
--compare_debug_mode 0 \
--task $task \
--split candidate \
--gpt2 $ptm_name \
--do_zeroshot \
--method $method \
--train_seed 100 \
--select_example_label_balance 1 \
--template_idx $template_idx \
--batch_size 48 \
--select_metric $progressive_select_metric \
--exp_tag $progressive_exp_tag \
--mask_same_candidate_indication_pair 1