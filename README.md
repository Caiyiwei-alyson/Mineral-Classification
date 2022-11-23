# mineral-classification
Codes for MIneral Classification based Deep Learning

Our Checkpoints and Data: 
Baidu Netdisk, Extract Code:

Data prepration:
Data root:
  -train
    - arsenopyrite
      - .jpg
      - .jpg
        ...
        - gold
      - ...
        ...
    -val
        - arsenopyrite
      - .jpg
      - .jpg
        ...
        - gold
      - ...
        ...
    -test
        - arsenopyrite
      - .jpg
      - .jpg
        ...
        - gold
      - ...
        ...

  Train:

```
python train.py --name " " --data_root "root path" --save_root_path "save_root_path" --val_save_root_path "val_save_root_path" --method "swin"
```

   Test:

```
python test.py --data_root "root path" --method "swin" --test_pretrain_weights "checkpoints path" --this_val_save_path "this_val_save_path"
```

