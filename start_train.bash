#!/bin/bash

data_dir="dataset/Dunkelbild/"
roi_info="dataset/roi_dunkelbild.txt"
epochs=100
batch_size=16
lr=0.001
seed=42
input_size=400
model="vgg16"
trainsplit=0.8
rotate=0.5
flip=0.5
zoom=0.5
shift=0.5
noise=0.5
squeeze=0.5

# Add more parameters as needed

python3 train.py --data_dir=$data_dir --roi_info=$roi_info --epochs=$epochs --batch_size=$batch_size --lr=$lr --seed=$seed \
--input_size=$input_size --model=$model --trainsplit=$trainsplit --rotate=$rotate --flip=$flip --zoom=$zoom --shift=$shift \
--noise=$noise --squeeze=$squeeze
# Add more parameters as needed
