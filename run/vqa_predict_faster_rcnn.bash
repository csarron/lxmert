# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=snap/vqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/vqa_predict.py \
    --tiny --valid ""  \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --optim bert --lr 5e-5 --epochs 1 \
    --tqdm --output $output ${@:3}

#
# bash run/vqa_predict.bash 0 vqa_lxr955_tiny-results --test minival --load snap/vqa/vqa_lxr955_tiny/BEST