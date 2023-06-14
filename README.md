# Evolved-Part-Masking-for-Self-Supervised-Learning
The code for the paper Evolved Part Masking for Self-Supervised Learning.
[https://openaccess.thecvf.com/content/CVPR2023/papers/Feng_Evolved_Part_Masking_for_Self-Supervised_Learning_CVPR_2023_paper.pdf](https://openaccess.thecvf.com/content/CVPR2023/papers/Feng_Evolved_Part_Masking_for_Self-Supervised_Learning_CVPR_2023_paper.pdf)

## Requirements
* [PyTorch](https://pytorch.org/) version >= 1.11.0

* Install other libraries via
```
pip install -r requirements.txt
```

## Pre-training

```
python Script/main_pretrain.py
```

## Fine-tune

```
python Script/main_finetune.py
```

## Linear probe

```
python Script/main_linprobe.py
```

## Citation
If you use our work, please cite:
```
@inproceedings{feng2023evolved,
  title={Evolved Part Masking for Self-Supervised Learning},
  author={Feng, Zhanzhou and Zhang, Shiliang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10386--10395},
  year={2023}
}
```
## Acknowledgement

The implementation relies on resources from [MAE](https://github.com/facebookresearch/mae). We thank the original authors for their open-sourcing.
