# gg-clip-vit
Week 4 Clip and ViT

## Tyrone's code

```bash
# local machine
# edit ~/.ssh/config to point 'mlx' at Computa
./send.sh
# computa
source ssh.sh
cd /workspace/gg-clip-vit
python run precompute_images.py
python run train_models.py
```