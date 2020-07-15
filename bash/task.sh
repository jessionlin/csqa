python ../task.py\
 --batch_size 8\
 --lr 1e-5\
 --num_train_epochs 10\
 --warmup_proportion 0.1\
 --weight_decay 0.1\
 --fp16 0\
 --train_file_name ../../conceptnet/9_rels/train_data.json\
 --devlp_file_name ../../conceptnet/9_rels/dev_data.json\
 --trial_file_name ../../conceptnet/9_rels/test_data.json\
 --pred_file_name  ../data/task_result.json\
 --output_model_dir ../data/conceptnet/9_rels\
 --bert_model_dir ../../albert-xxlarge-v2\
 --bert_vocab_dir ../../albert-xxlarge-v2\
 --print_step 100\
 --mission train
 