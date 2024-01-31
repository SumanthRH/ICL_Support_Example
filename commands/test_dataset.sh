diversity_score_scale=4
method=direct
ptm_name=hf-internal-testing/tiny-random-gpt2
template_idx=0
direct_plus=0

task=$1 \
template_idx=$template_idx \
method=$method \
progressive_p=2 \
initial_indication_set_size=14 \
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
ptm_name=$ptm_name \
final_candidate_size=500 \
label_balance=1 \
candidate_example_num_total=-1 \
candidate_example_num_every_label=4 \
direct_plus=$direct_plus \
diversity_score_scale=$diversity_score_scale \
bash commands/run_search.sh