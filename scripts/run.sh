dataset=face
factor=2
sfx=.jpg

edit=clown
scale=0.8

prompt="Turn him into a clown"
guidance_scale=7.5
image_guidance_scale=1.5

################ 1. Run an initial NeRF for getting the depths  #################

# rm -rf lama/LaMa_test_images/*
# rm -rf lama/output/label/*

# python DS_NeRF/run_nerf.py \
# --config DS_NeRF/configs/config.txt \
# --render_factor 1 \
# --prepare \
# --i_weight 4000 \
# --i_video 4000 \
# --i_feat 4000 \
# --N_iters 4001 \
# --expname $dataset \
# --datadir ./data/$dataset \
# --factor $factor \
# --N_gt 0

##################################################################################

cd lama
export TORCH_HOME=$(pwd)
export PYTHONPATH=$(pwd)

########### 2. Run LaMa to generate geometry and appearance guidance  ############

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
# cp ../data/$dataset/images_$factor/*$sfx LaMa_test_images
# mkdir LaMa_test_images/label
# cp ../data/$dataset/images_$factor/label/*.png LaMa_test_images/label

# python bin/predict.py \
# refine=True \
# model.path=$(pwd)/big-lama \
# indir=$(pwd)/LaMa_test_images \
# outdir=$(pwd)/output \
# dataset.img_suffix=$sfx

# rm -r ../data/$dataset/images_$factor/lama_images
# mkdir ../data/$dataset/images_$factor/lama_images
# mkdir ../data/$dataset/images_$factor/lama_images/label
# cp ./output/label/*.png ../data/$dataset/images_$factor/lama_images
# cp ../data/$dataset/images_$factor/label/*.png ../data/$dataset/images_$factor/lama_images/label

##################################################################################

cd ..
export TORCH_HOME=$(pwd)
export PYTHONPATH=$(pwd)

########################## 3. Run multiview inpainter  ###########################

# python DS_NeRF/run_nerf.py \
# --config DS_NeRF/configs/config.txt \
# --i_feat 4000 \
# --lpips \
# --i_weight 4000 \
# --i_video 4000 \
# --N_iters 4001 \
# --expname ${dataset}_scene \
# --datadir ./data/$dataset \
# --N_gt 0 \
# --factor $factor

##################################################################################



############################# 4. Run Segmented NeRF  #############################

# python DS_NeRF/run_nerf.py \
# --config DS_NeRF/configs/config.txt \
# --render_factor 1 \
# --prepare \
# --i_weight 4000 \
# --i_video 4000 \
# --i_feat 4000 \
# --N_iters 4001 \
# --expname ${dataset}_object \
# --datadir ./data/$dataset \
# --factor $factor \
# --N_gt 0 \
# --segmented_NeRF

##################################################################################



########################### 5. Run Instruct-NeRF2NeRF  ###########################

# python DS_NeRF/run_nerf.py \
# --config DS_NeRF/configs/config.txt \
# --render_factor 1 \
# --prepare \
# --i_weight 6000 \
# --i_video 6000 \
# --i_feat 6000 \
# --N_iters 6001 \
# --expname ${dataset}_object_$edit \
# --base_expname ${dataset}_object \
# --datadir ./data/$dataset \
# --factor $factor \
# --N_gt 0 \
# --in2n \
# --prompt $prompt \
# --guidance_scale $guidance_scale \
# --image_guidance_scale $image_guidance_scale \
# --lpips \
# --lpips_lambda 0.1 \
# --segmented_NeRF

##################################################################################



############################## 5. Merge Two Scenes  ##############################

# python DS_NeRF/merge_nerf.py \
# --config DS_NeRF/configs/config.txt \
# --render_factor 1 \
# --expname ${dataset}_merged_$edit \
# --object_expname ${dataset}_object_$edit \
# --scene_expname ${dataset}_scene \
# --datadir ./data/$dataset \
# --factor $factor \
# --scale $scale

##################################################################################