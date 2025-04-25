
CUDA_VISIBLE_DEVICES=0,1 PORT=29508 nohup ./tools/dist_train.sh /data01/public_dataset/xu/project/Boost/configs/cityscapes/faster-rcnn_r50_fpn_1x_cityscapes_c_boost.py 2  >cityscapes_c_boost.log 2>&1&


