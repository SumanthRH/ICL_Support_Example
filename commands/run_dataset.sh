echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
diversity_score_scale=4
template_idx=0

method=direct

direct_plus=0

ptm_name=gpt2-large

batch_size=16 \
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
bash commands/run_filter_and_search.sh

done
done
done
done