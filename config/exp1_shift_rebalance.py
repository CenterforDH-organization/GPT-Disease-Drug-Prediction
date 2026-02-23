# Experiment 1: SHIFT imbalance re-training
# Goal: Improve minority SHIFT classes (Decrease/Increase)

model_type = 'composite'

# SHIFT loss/sampling
shift_loss_type = 'focal'                 # 'focal' or 'ce'
shift_focal_gamma = 2.0
shift_auto_class_weight_mode = 'sqrt_inverse'
shift_class_weights = []                  # keep empty to auto-compute
shift_class_weights_from_drug_tokens_only = True

shift_sampling_strategy = 'weighted'      # 'uniform' or 'weighted'
shift_sampling_boost_factor = 2.0
shift_sampling_minority_classes = [1, 3]  # raw SHIFT ids
shift_sampling_classes_are_shifted = False
shift_sampling_drug_only = True

# Emphasize SHIFT objective
loss_weight_shift = 20.0

