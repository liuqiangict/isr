

```shell
sudo pip install -r requirements.txt

export PYTHONPATH=/path/to/turingISR/src/models

# scale 2
python inference.py \
    --eval_datadir=/path/to/evaluate/images/ \
    --output_dir=/output/folder/ \
    --arch=drln \
    --pretrained=/path/to/pretrained/cehckpoints/ \
    --evaluate \
    --scale=2 \
    --n_resblocks=6 \
    --n_feats=64 \
    --gpu=0

```
