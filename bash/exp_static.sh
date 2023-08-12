# Under current path, run this:
# $ bash exp_static.sh RandomForest 1 5
# the example means to use model KNN, use test with external testset, with window_siye=250

cd .. # go back to root path

model="$1" # which model to train
ext="$2" # 0: not use external testset, 1: use external testset
wl="$3" # window size

python train.py \
    --train_split_method_paths "dataset/chor2_20230609/split_method.yaml" \
    --trainset_path "dataset/chor2_20230609/unknown.NoHead.csv" \
    --test_split_method_paths "dataset/testset_20230627/split_method.yaml" \
    --testset_path "dataset/testset_20230627/unknown.NoHead.csv" \
    --split_ratio 0.9 \
    --window_size $wl \
    --model $model \
    --max_depth 6 \
    --random_state 0 \
    --n_neighbor 20 \
    --exp_group "Static" \
    --outside_test $ext \
    --save_res 1
