# Under current path, run this:
# $ bash exp_agree.sh RandomForest 1 100
# the example means to use model KNN, use test with external testset, with window_siye=250

cd .. # go back to root path

model="$1" # which model to train
ext="$2" # 0: not use external testset, 1: use external testset
wl="$3" # window size

python train.py \
    --exp_group "Agree" \
    --split_ratio 0.9 \
    --window_size $wl \
    --outside_test $ext \
    --save_res 1 \
    --model $model \
    --n_neighbor 20 \
    --max_depth 6 \
    --random_state 0