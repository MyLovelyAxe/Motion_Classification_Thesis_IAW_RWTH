# Under current path, run this:
# $ bash exp_static.sh KNN 1
# the example means to use model KNN, use test with external testset, with window_siye=250

cd .. # go back to root path

model="$1" # which model to train
ext="$2" # 0: not use external testset, 1: use external testset

if [[ $model == "KNN" ]]; then

    python train_static_ML.py \
        --train_split_method_paths "dataset/chor2_20230609/split_method.yaml" \
        --trainset_path "dataset/chor2_20230609/x_data_UpperLowerBody.npy" \
                        "dataset/chor2_20230609/y_data_UpperLowerBody.npy" \
        --test_split_method_paths "dataset/testset_20230627/split_method.yaml" \
        --testset_path "dataset/testset_20230627/x_data_UpperLowerBody.npy" \
                       "dataset/testset_20230627/y_data_UpperLowerBody.npy" \
        --model "KNN" \
        --n_neighbor 20 \
        --exp_group "Static" \
        --outside_test $ext \
        --save_res 1

elif [[ $model == "RandomForest" ]]; then

    python train_static_ML.py \
        --train_split_method_paths "dataset/chor2_20230609/split_method.yaml" \
        --trainset_path "dataset/chor2_20230609/x_data_UpperLowerBody.npy" \
                        "dataset/chor2_20230609/y_data_UpperLowerBody.npy" \
        --test_split_method_paths "dataset/testset_20230627/split_method.yaml" \
        --testset_path "dataset/testset_20230627/x_data_UpperLowerBody.npy" \
                       "dataset/testset_20230627/y_data_UpperLowerBody.npy" \
        --model "RandomForest" \
        --max_depth 6 \
        --random_state 0 \
        --exp_group "Static" \
        --outside_test $ext \
        --save_res 1

elif [[ $model == "SVM" ]]; then

    python train_static_ML.py \
        --train_split_method_paths "dataset/chor2_20230609/split_method.yaml" \
        --trainset_path "dataset/chor2_20230609/x_data_UpperLowerBody.npy" \
                        "dataset/chor2_20230609/y_data_UpperLowerBody.npy" \
        --test_split_method_paths "dataset/testset_20230627/split_method.yaml" \
        --testset_path "dataset/testset_20230627/x_data_UpperLowerBody.npy" \
                       "dataset/testset_20230627/y_data_UpperLowerBody.npy" \
        --model "SVM" \
        --exp_group "Static" \
        --outside_test $ext \
        --save_res 1

fi
