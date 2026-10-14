---
title: The EYE live detector
emoji: 👁️
colorFrom: green
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
short_description: Finds 73 everyday objects in any uploaded image
---

# The EYE · live detector

IE7615 Deep Learning, Group 7 (Milestone 3). Upload an image and the selected detector, **yolo26s_boost**
(YOLO26-s, 1,280 px), finds the course's 73 objects in it.

Same rules as The EYE v3 for uploads, all chosen on the validation set:

| Image | Threshold | Post-processing |
|---|---|---|
| Grid (sides multiples of 224 px) | 0.10 | one box per 224-px cell |
| Scene (anything else) | 0.35 | overlapping boxes of any name reduced to the most confident (IoU 0.5) |
| One object per photo (optional) | 0.10 | photo resized to 224 x 224 like the dataset photos |

Code and results: https://github.com/tanmayi123/Discriminative-Project-Deep-Learning/tree/main/Milestone_3
