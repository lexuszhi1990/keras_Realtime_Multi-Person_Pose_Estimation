### setup on 172
make clean && make
bash scripts/demo.sh

docker run --network host --rm -it -v /home/david/keras_Realtime_Multi-Person_Pose_Estimation:/app-dev -v /data/david/cocoapi:/mnt/data/coco -v /data/david/models/keras_openpose:/mnt/models -v /data/david/logdir/kares-openpose:/mnt/logs keras-openpose:latest

demo:
`CUDA_VISIBLE_DEVICES=7 python3 demo_image.py --image sample_images/ski.jpg`

build image:
`python3 training/coco_masks_hdf5.py`

train:
`CUDA_VISIBLE_DEVICES=7 python3 train_pose.py --task=train --datapath=/mnt/data/coco --modelpath=/mnt/models --logpath=/mnt/logs --batch_size=4`

### packages

apt-get install -y python3.6-dev
pip3 install opencv-python easydict cython configobj h5py

