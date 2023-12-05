#!/bin/bash
for channels in "16 32 64 128" "32 64 128 256" "64 128 256 512"
do
  for dropout in "0.1" "0.3" "0.5"
  do
    for learning_rate in "0.001" "0.0001"
    do
      ./shell/training/knee_cv/groot_train_segan_3d_cv.sh "$channels" "$dropout" "$learning_rate" &>> segan_fromscratch.log
    done
  done
done
for dropout in "0.1" "0.3" "0.5"
do
  for learning_rate in "0.001" "0.0001"
  do
    ./shell/training/knee_cv/groot_train_segresnetvae_3d_cv.sh "$dropout" "$learning_rate" &>> segresnetvae_fromscratch.log
  done
done
for channels in "16 32 64 128" "32 64 128 256" "64 128 256 512"
do
  for dropout in "0.1" "0.3" "0.5"
  do
    for learning_rate in "0.001" "0.0001"
    do
      ./shell/training/knee_cv/groot_train_unet_3d_cv.sh "$channels" "$dropout" "$learning_rate" &>> unet_fromscratch.log
    done
  done
done
for channels in "16 16 32 64 32 16" "32 32 64 128 64 32"
do
  for dropout in "0.1" "0.3" "0.5"
  do
    for learning_rate in "0.001" "0.0001"
    do
      ./shell/training/knee_cv/groot_train_unetpp_3d_cv.sh "$channels" "$dropout" "$learning_rate" &>> unetpp_fromscratch.log
    done
  done
done
for dropout in "0.1" "0.3" "0.5"
do
  for learning_rate in "0.001" "0.0001"
  do
    ./shell/training/knee_cv/groot_train_unetr_3d_cv.sh "$dropout" "$learning_rate" &>> unetr_fromscratch.log
  done
done