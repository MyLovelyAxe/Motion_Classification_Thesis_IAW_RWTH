# Under current path, run this:
# $ bash exp_agree.sh KNN 1 100
# the example means to use model KNN, use test with external testset, with window_siye=250

cd .. # go back to root path

model="$1" # which model to train
ext="$2" # 0: not use external testset, 1: use external testset
wl="$3" # window size

python train.py \
    --train_split_method_paths "dataset/agree_20230801/split_method.yaml" \
    --trainset_path "dataset/agree_20230801/x_data_UpperLowerBody.npy" \
                    "dataset/agree_20230801/y_data_UpperLowerBody.npy" \
    --test_split_method_paths "dataset/agree_test_20230801/split_method.yaml" \
    --testset_path "dataset/agree_test_20230801/unknown.NoHead.csv" \
    --split_ratio 0.9 \
    --window_size $wl \
    --model $model \
    --max_depth 6 \
    --random_state 0 \
    --n_neighbor 20 \
    --exp_group "Agree" \
    --outside_test $ext \
    --save_res 1