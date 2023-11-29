if [ -z $DATA ]; then
    DATA=custom
fi

if [ -z $VIDEO ]; then
    VIDEO=custom.mp4
fi

if [ -z $H ]; then
    H=0
fi

if [ -z $W ]; then
    W=0
fi

if [ -z $NUM_DATA ]; then
    NUM_DATA=100
fi

if [ -z $FACTOR ]; then
    FACTOR=2
fi

if [ ! -f "data/$DATA/original.mp4" ]; then
    if [ ! -f $VIDEO ]; then
        echo "$VIDEO doesn't exists"
        exit 1
    fi
fi

if [ ! -d "data/$DATA" ]; then
    mkdir -p data/$DATA
fi



################## 1. Copy Original Video to the data directory ##################
if [ ! -f "data/$DATA/original.mp4" ]; then
    echo "################## 1. Copy Original Video to the data directory ##################"
    cp $VIDEO data/$DATA/original.mp4
    echo "##################################################################################"
fi
##################################################################################


####################### 2. Unpack all frames of the video ########################
if [ ! -d "data/$DATA/original_images" ]; then
    echo "####################### 2. Unpack all frames of the video ########################"
    mkdir -p data/$DATA/temp_original_images
    ffmpeg -i data/$DATA/original.mp4 data/$DATA/temp_original_images/%04d.png
    mv data/$DATA/temp_original_images data/$DATA/original_images
    echo "##################################################################################"
fi
##################################################################################


################# 3. Uniformly Sample, Crop and Resize the Image #################
if [ ! -d "data/$DATA/images" ]; then
    echo "################# 3. Uniformly Sample, Crop and Resize the Image #################"
    mkdir -p data/$DATA/temp_images
    python scripts/crop_and_resize.py \
    --src_root data/$DATA/original_images \
    --dst_root data/$DATA/temp_images \
    --H $H \
    --W $W \
    --num_data $NUM_DATA
    mv data/$DATA/temp_images data/$DATA/images
    echo "##################################################################################"
fi
##################################################################################


################################# 4. Run COLMAP ##################################
if [ ! -f "data/$DATA/poses_bounds.npy" ]; then
    echo "################################# 4. Run COLMAP ##################################"
    python imgs2poses.py --data_dir data/$DATA
    echo "##################################################################################"
fi
##################################################################################


########################## 5. Interactive Segmentation ###########################
if [ ! -d "data/$DATA/images/label" ]; then
    echo "########################## 5. Interactive Segmentation ###########################"
    python InteractiveSegmentator.py --data_root ./data/$DATA/images
    echo "##################################################################################"
fi
##################################################################################


################################ 6. Downsize Data ################################
if [ ! -d "data/$DATA/images_$FACTOR" ]; then
    echo "################################ 6. Downsize Data ################################"
    python scripts/downsize.py --data_root ./data/$DATA
    echo "##################################################################################"
fi
##################################################################################