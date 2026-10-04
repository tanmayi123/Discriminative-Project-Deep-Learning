# Milestone 1 — Results summary

## Dataset
- 73 classes (Object IDs), 7343 files in the merged dataset → 6942 images used.
- Removed: 385 background-only (filename tag), 16 byte-identical duplicates, 0 unreadable.
- Split 70/15/15 stratified per Object ID, seed 42: 4859 train / 1041 val / 1042 test (data signature 6ed2c1673798).
- Out-of-fold linear-probe accuracy during the data audit: 0.983.
- Near-duplicate test images (cosine ≥ 0.92 to a same-class train image): 99 (9.5%).

## Model comparison (sorted by validation macro-F1)
```
                experiment                                                        approach  val_macro_F1  test_acc         95% CI  test_macro_F1  test_acc_hard_subset  test_acc_TTA  params_M  trainable_M  train_min  ms_per_img  best_epoch
D_efficientnet_b0_finetune                     EfficientNet-B0 (ImageNet) fully fine-tuned        0.9923    0.9818 [0.973, 0.989]         0.9815                0.9799        0.9856    4.1011       4.1011    26.2252      0.4708          15
       C_resnet50_finetune                           ResNet-50 (ImageNet) fully fine-tuned        0.9914    0.9866 [0.979, 0.993]         0.9863                0.9852        0.9885   23.6576      23.6576    26.3097      0.5599          14
  E_convnext_tiny_finetune                       ConvNeXt-Tiny (ImageNet) fully fine-tuned        0.9903    0.9942 [0.989, 0.998]         0.9942                0.9936        0.9933   27.8763      27.8763    14.8680      0.7902          16
 F_convnext_tiny_basic_aug Ablation: ConvNeXt-Tiny fine-tuned with basic augmentation only        0.9883    0.9866 [0.979, 0.993]         0.9865                0.9852        0.9875   27.8763      27.8763     6.2100      0.7706           5
   B_convnext_linear_probe                     Frozen ImageNet ConvNeXt-Tiny + linear head        0.9748    0.9741 [0.964, 0.984]         0.9740                0.9714        0.9818   27.8763       0.0561     4.9103      0.8032          12
      A_custom_cnn_scratch                         Plain 10-layer CNN trained from scratch        0.8880    0.8858 [0.867, 0.904]         0.8827                0.8738        0.9050    3.4147       3.4147    57.3137      0.3821          57
```

## Robustness (test accuracy under simulated conditions)
```
                experiment  clean  small_object  cropped  occluded  low_quality  mean_drop
      A_custom_cnn_scratch 0.8858        0.7591   0.8772    0.7370       0.2678     0.2255
   B_convnext_linear_probe 0.9741        0.8570   0.9693    0.8935       0.5058     0.1677
       C_resnet50_finetune 0.9866        0.9559   0.9798    0.9165       0.4184     0.1689
D_efficientnet_b0_finetune 0.9818        0.9511   0.9818    0.9031       0.4779     0.1533
  E_convnext_tiny_finetune 0.9942        0.9789   0.9904    0.9155       0.5701     0.1305
 F_convnext_tiny_basic_aug 0.9866        0.8656   0.9798    0.8839       0.3666     0.2126
```

## Significance (McNemar exact test vs. selected model)
```
                        vs  only_best_right  only_other_right  p_value  significant (p<0.05)
      A_custom_cnn_scratch              114                 1   0.0000                  True
   B_convnext_linear_probe               24                 3   0.0000                  True
       C_resnet50_finetune                8                 0   0.0078                  True
D_efficientnet_b0_finetune               13                 0   0.0002                  True
 F_convnext_tiny_basic_aug                9                 1   0.0215                  True
```

## Selection on validation data (clean + perturbed)
```
                experiment  val_macro_F1  val_small_object  val_cropped  val_occluded  val_low_quality  val_robust_mean  params_M
  E_convnext_tiny_finetune        0.9903            0.9712       0.9875        0.8934           0.5389           0.8477   27.8763
D_efficientnet_b0_finetune        0.9923            0.9481       0.9837        0.8866           0.4592           0.8194    4.1011
       C_resnet50_finetune        0.9914            0.9549       0.9866        0.8895           0.3967           0.8069   23.6576
 F_convnext_tiny_basic_aug        0.9883            0.8511       0.9808        0.8473           0.3794           0.7646   27.8763
```

## Selected model
Selection rule: among models within 0.5 pt of the best validation macro-F1 (D_efficientnet_b0_finetune, C_resnet50_finetune, E_convnext_tiny_finetune, F_convnext_tiny_basic_aug), choose the highest mean accuracy on perturbed VALIDATION images (size as final tie-break). Selected: E_convnext_tiny_finetune. (Size alone would have chosen D_efficientnet_b0_finetune.) The test set was not used for selection.
**E_convnext_tiny_finetune** — ConvNeXt-Tiny (ImageNet) fully fine-tuned. Validation macro-F1 0.9903; test accuracy 0.9942 [0.989, 0.998]; test macro-F1 0.9942; hard-subset accuracy 0.9936; with TTA 0.9933; 27.9M parameters; 0.79 ms/image inference (warm GPU, batch 32).

## Background-shortcut test
On 385 excluded background-only images (no object visible), the selected model predicted the photographer's own Object ID 32.5% of the time (chance = 1.4%), mean confidence 0.36. This indicates substantial reliance on background cues.