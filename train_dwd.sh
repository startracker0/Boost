

CUDA_VISIBLE_DEVICES=0,1 PORT=29501 nohup ./tools/dist_train.sh /data01/public_dataset/xu/project/Boost/configs/dwd/faster-rcnn_r101_caffe_20e_dwd_boost.py 2  >dwd_boost.log 2>&1&

