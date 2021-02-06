

```shell
sudo pip install -r requirements.txt

export PYTHONPATH=/path/to/turingISR/src/models

# scale 2
python inference.py \
    --eval_datadir=/path/to/inference/images/ \
    --output_dir=/output/folder/ \
    --arch=drln \
    --pretrained=/path/to/your/checkpoint/
    --inference \
    --scale=2 \
    --n_resblocks=6 \
    --n_feats=64 \
    --gpu=0
```
