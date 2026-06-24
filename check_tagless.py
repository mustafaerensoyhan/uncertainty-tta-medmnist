import numpy as np
from src.config import get_config
from src.model import build_model
from src.data import get_dataloaders
from src.train import predict_probs
from src.metrics import expected_calibration_error as ece
from src.utils import get_device, load_checkpoint, checkpoint_filename

d = get_device()
for ds in ['pathmnist','dermamnist','pneumoniamnist','breastmnist','bloodmnist','organamnist']:
    c = get_config(ds)
    _, _, tl, _ = get_dataloaders(ds, batch_size=64, img_size=64, num_workers=0, root='./data')
    out = {}
    for tag in ['', '_seed0']:
        m = build_model('resnet18', num_classes=c.n_classes, pretrained=False)
        load_checkpoint(m, 'checkpoints/' + checkpoint_filename(ds, 'resnet18', tag), device=d)
        m.to(d)
        p, y = predict_probs(m, tl, d)
        out[tag or 'tagless'] = float(ece(p, y, 10))
    flag = '  <-- MISMATCH' if abs(out['tagless'] - out['_seed0']) > 0.01 else ''
    print(f"{ds:14s} tagless={out['tagless']:.4f}  seed0={out['_seed0']:.4f}{flag}")
