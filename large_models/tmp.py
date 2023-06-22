def permute_string(s):
    sn = ''
    for i in range(6):
        sn += s.replace('--train_set_seed 1', f'--train_set_seed {i + 1}')
        sn += '\n'
        sn += 'sleep 45'
        sn += '\n'
    return sn


print(permute_string(
    'nohup python large_models/run.py  --num_train 16 --num_eval 1000 --logging_steps 10 --trainer regular --learning_rate 1e-5 --num_train_epochs 5 --per_device_train_batch_size 8 --evaluation_strategy epoch --save_total_limit 0 --train_as_classification --task_name BoolQ --output_dir result/RTE--ft-5-8-1e-5-0 --in_context_fine_tune  --in_context_fine_tune_v2 --permutations_per_example 4 --train_set_seed 1  &'))
