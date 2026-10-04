# IE7615 Milestone 1 - results summary

## Data
- Indexed images: 7343 in 73 folders; junk entries skipped: 2; corrupt: 0
- Background-only images removed: 385 (filename tags _bg_ / BG###; OBJ004 has no tags, so its background shots stay in)
- Exact duplicates removed from object photos (MD5, after background removal): 16 same-class + 0 cross-class
- Final: 6942 images, 73 classes, 89-106 per class
- Split 70/15/15 stratified, seed 42: train 4859 / val 1041 / test 1042
- Label audit (5-fold out-of-fold): 97.9% ; suspects: 50
- Near-duplicate test images (cos >= 0.92): 99; hard subset: 943
- Class-weighted loss: on (train ratio 1.19)

Note: the pretrained models were shortlisted from earlier experiments (Tanmayi's and Santhosh's runs); the final model is chosen on validation data only.

## Models
| model             | description                                       |   params_M |   trainable_M |   epochs |   batch |   lr_head |   lr_backbone | aug    |
|:------------------|:--------------------------------------------------|-----------:|--------------:|---------:|--------:|----------:|--------------:|:-------|
| A2_CustomCNN_w384 | Custom CNN (Santhosh): w384, Kaiming, 60 ep, bs32 |       3.41 |         3.415 |       60 |      32 |     0.002 |        0.002  | robust |
| C_ResNet50        | ResNet-50 fine-tuned                              |      23.66 |        23.658 |       20 |      32 |     0.001 |        0.0001 | robust |
| D_EfficientNetB0  | EfficientNet-B0 fine-tuned                        |       4.1  |         4.101 |       20 |      32 |     0.001 |        0.0003 | robust |
| E_ConvNeXtT       | ConvNeXt-Tiny fine-tuned                          |      27.88 |        27.876 |       20 |      32 |     0.001 |        0.0001 | robust |
| G_DenseNet121     | DenseNet-121 fine-tuned                           |       7.03 |         7.029 |       20 |      32 |     0.001 |        0.0001 | robust |

## Main comparison (test set)
| model             |   val_f1 |   test_acc | acc_95CI     |   test_f1 |   hard_acc |   top5_acc |   tta_acc |   test_robust_mean |   test_errors |   params_M |   train_min |   ms_per_img |
|:------------------|---------:|-----------:|:-------------|----------:|-----------:|-----------:|----------:|-------------------:|--------------:|-----------:|------------:|-------------:|
| A2_CustomCNN_w384 |    89.56 |      89.06 | [87.1, 91.1] |     88.72 |      87.91 |      97.12 |     91.17 |              66.96 |           114 |       3.41 |       30.58 |         0.77 |
| C_ResNet50        |    99.22 |      98.66 | [98.0, 99.3] |     98.65 |      98.52 |      99.71 |     98.85 |              78.96 |            14 |      23.66 |       11.43 |         1.26 |
| D_EfficientNetB0  |    99.41 |      98.75 | [98.1, 99.3] |     98.74 |      98.62 |      99.81 |     98.94 |              81.81 |            13 |       4.1  |       11.71 |         0.7  |
| E_ConvNeXtT       |    99.61 |      99.23 | [98.7, 99.7] |     99.23 |      99.15 |      99.81 |     99.23 |              84.74 |             8 |      27.88 |       13.4  |         1.43 |
| G_DenseNet121     |    99.13 |      98.66 | [97.9, 99.3] |     98.63 |      98.52 |      99.9  |     98.85 |              73.59 |            14 |       7.03 |       14.09 |         1.51 |

## Selection (validation only)
| model            |   val_f1 |   val_robust_mean |   params_M |
|:-----------------|---------:|------------------:|-----------:|
| E_ConvNeXtT      |    99.61 |             83.98 |      27.88 |
| D_EfficientNetB0 |    99.41 |             81.63 |       4.1  |
| C_ResNet50       |    99.22 |             79.04 |      23.66 |
| G_DenseNet121    |    99.13 |             73.87 |       7.03 |

**Selected model: E_ConvNeXtT** - rule: within 0.5 pt of best val macro-F1 -> highest perturbed-val accuracy -> fewest params.

## McNemar exact test vs selected model
| compared_with     |   only_selected_correct |   only_other_correct |   p_value | significant   |
|:------------------|------------------------:|---------------------:|----------:|:--------------|
| A2_CustomCNN_w384 |                     110 |                    4 |    0      | yes           |
| C_ResNet50        |                      11 |                    5 |    0.2101 | no            |
| D_EfficientNetB0  |                      10 |                    5 |    0.3018 | no            |
| G_DenseNet121     |                       9 |                    3 |    0.146  | no            |

## Robustness (test accuracy %)
|                   |   clean |   small_object |   cropped |   occluded |   low_quality |   robust_mean |   mean_drop |
|:------------------|--------:|---------------:|----------:|-----------:|--------------:|--------------:|------------:|
| A2_CustomCNN_w384 |   89.06 |          76.68 |     90.12 |      74.95 |         26.1  |         66.96 |       22.1  |
| C_ResNet50        |   98.66 |          95.2  |     98.08 |      89.64 |         32.92 |         78.96 |       19.7  |
| D_EfficientNetB0  |   98.75 |          94.82 |     97.98 |      89.44 |         45.01 |         81.81 |       16.94 |
| E_ConvNeXtT       |   99.23 |          97.6  |     98.46 |      91.17 |         51.73 |         84.74 |       14.49 |
| G_DenseNet121     |   98.66 |          93.47 |     98.18 |      82.82 |         19.87 |         73.59 |       25.07 |

## Background-shortcut test
| model             |   n_bg_images |   own_id_predicted_pct |   chance_pct |   mean_conf_on_bg |   mean_conf_on_test_objects |
|:------------------|--------------:|-----------------------:|-------------:|------------------:|----------------------------:|
| A2_CustomCNN_w384 |           378 |                 33.069 |         1.37 |             0.377 |                       0.799 |
| C_ResNet50        |           378 |                 36.508 |         1.37 |             0.346 |                       0.867 |
| D_EfficientNetB0  |           378 |                 38.095 |         1.37 |             0.376 |                       0.886 |
| E_ConvNeXtT       |           378 |                 35.714 |         1.37 |             0.356 |                       0.89  |
| G_DenseNet121     |           378 |                 34.656 |         1.37 |             0.332 |                       0.867 |

## Mistakes per model
| model             |   test |   val |
|:------------------|-------:|------:|
| A2_CustomCNN_w384 |    114 |   106 |
| C_ResNet50        |     14 |     8 |
| D_EfficientNetB0  |     13 |     6 |
| E_ConvNeXtT       |      8 |     4 |
| G_DenseNet121     |     14 |     9 |

## Weakest classes (selected model)
|               |   precision |   recall |   f1-score |   support |
|:--------------|------------:|---------:|-----------:|----------:|
| images_OBJ061 |       0.929 |    0.929 |      0.929 |        14 |
| images_OBJ027 |       0.875 |    1     |      0.933 |        14 |
| images_OBJ054 |       0.933 |    0.933 |      0.933 |        15 |
| images_OBJ014 |       1     |    0.929 |      0.963 |        14 |
| images_OBJ042 |       1     |    0.929 |      0.963 |        14 |
| images_OBJ037 |       1     |    0.929 |      0.963 |        14 |
| images_OBJ012 |       1     |    0.929 |      0.963 |        14 |
| images_OBJ063 |       1     |    0.929 |      0.963 |        14 |
| images_OBJ059 |       1     |    0.929 |      0.963 |        14 |
| images_OBJ045 |       0.933 |    1     |      0.966 |        14 |