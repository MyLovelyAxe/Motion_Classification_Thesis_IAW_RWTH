# Under current path, run this:
# $ bash exp_dynamic.sh KNN 1 200
# the example means to use model KNN, use test with external testset, with window_siye=250

cd .. # go back to root path

model="$1" # which model to train
ext="$2" # 0: not use external testset, 1: use external testset
wl="$3" # window size

python train.py \
    --train_split_method_paths "dataset/dynamic1_20230706/split_method.yaml" \
                                "dataset/dynamic2_20230706/split_method.yaml" \
                                "dataset/dynamic3_20230706/split_method.yaml" \
    --trainset_path "dataset/dynamic_dataset/x_data_UpperLowerBody.npy" \
                    "dataset/dynamic_dataset/y_data_UpperLowerBody.npy" \
    --test_split_method_paths "dataset/dynamic_test_20230801/split_method.yaml" \
    --testset_path "dataset/dynamic_test_20230801/unknown.NoHead.csv" \
    --split_ratio 0.8 \
    --window_size $wl \
    --model $model \
    --max_depth 6 \
    --random_state 0 \
    --n_neighbor 20 \
    --exp_group "Dynamic" \
    --outside_test $ext \
    --save_res 1
