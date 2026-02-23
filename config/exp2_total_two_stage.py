# Experiment 2: TOTAL two-stage objective
# Goal: Improve skewed TOTAL target modeling

model_type = 'composite'

# TOTAL loss
total_loss_mode = 'two_stage'       # 'log_huber', 'two_stage', 'raw_mse'
total_two_stage_use_log = True      # positive regression in log1p space
total_huber_delta = 1.0
total_two_stage_cls_weight = 1.0    # BCE (0 vs non-zero)
total_two_stage_reg_weight = 2.0    # positive-only regression

# Keep TOTAL emphasized in multi-task training
loss_weight_total = 100.0

