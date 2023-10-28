dataset=bear
factor=2


# python imgs2poses.py data/$dataset

python InteractiveSegmentator.py --data_root ./data/$dataset/images_$factor

# python DS_NeRF/run_nerf.py \
# --config DS_NeRF/configs/config.txt \
# --render_factor 1 \
# --prepare \
# --i_weight 1000000000 \
# --i_video 1000000000 \
# --i_feat 4000 \
# --N_iters 4001 \
# --expname $dataset \
# --datadir ./data/$dataset \
# --factor $factor \
# --N_gt 0


# cd lama
# export TORCH_HOME=$(pwd)
# export PYTHONPATH=$(pwd)


# python bin/predict.py \
# refine=True \
# model.path=$(pwd)/big-lama \
# indir=$(pwd)/LaMa_test_images \
# outdir=$(pwd)/output


# rm -r ../data/$dataset/images_$factor/depth
# mkdir ../data/$dataset/images_$factor/depth
# cp ./output/label/*.png ../data/$dataset/images_$factor/depth


# rm -r LaMa_test_images/*
# rm -r output/label/*
# cp ../data/$dataset/images_$factor/*.png LaMa_test_images
# mkdir LaMa_test_images/label
# cp ../data/$dataset/images_$factor/label/*.png LaMa_test_images/label


# python bin/predict.py \
# refine=True \
# model.path=$(pwd)/big-lama \
# indir=$(pwd)/LaMa_test_images \
# outdir=$(pwd)/output


# rm -r ../data/$dataset/images_$factor/lama_images
# mkdir ../data/$dataset/images_$factor/lama_images
# cp ../data/$dataset/images_$factor/*.png ../data/$dataset/images_$factor/lama_images
# cp ./output/label/*.png ../data/$dataset/images_$factor/lama_images
# cd ..


# python DS_NeRF/run_nerf.py \
# --config DS_NeRF/configs/config.txt \
# --i_feat 200 \
# --lpips \
# --i_weight 1000000000000 \
# --i_video 1000 \
# --N_iters 10001 \
# --expname $data \
# --datadir ./data/$data \
# --N_gt 0 \
# --factor $factor