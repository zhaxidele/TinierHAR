# TinierHAR


Here you can find the code of the TinierHAR model (our Submission to Ubicomp/ISWC 2025) and other models commonly used in the efficient computing of HAR

Submission Title:   TinierHAR: Towards Ultra-Lightweight Deep Learning Models for Efficient Human Activity Recognition on Edge Devices

Note: We gratefully acknowledge the original code contributions from the TECO Group lead by [Prof. Micheal Beigl], which significantly supported the development of this work:
https://github.com/teco-kit/ISWC22-HAR



## TinierHAR

<img width="950" alt="Arch_TinyHAR++" src="https://github.com/user-attachments/assets/3bad2af1-6676-4971-a159-adb79974686d" />


## Data
The evaluated 14 datasets can be downloaded using the links provided in the [datasets](https://anonymous.4open.science/r/TinierHAR-B2F3/datasets) directory. 

# Training & Evaluation

```
python3 train.py --seeds [seed number] --model [model] --dataset [datasets]
```
e.g:

```
python3 train.py --seeds 5 --model tinierhar --dataset RecGym
```

The training and validation result will be stored in the directories of [output](https://anonymous.4open.science/r/TinierHAR-B2F3/output) and [Run_logs](https://anonymous.4open.science/r/TinierHAR-B2F3/Run_logs).


## Performance (TinierHAR vs. TinyHAR & DeepConvLSTM)

![Result](https://github.com/user-attachments/assets/8d474b2f-d8f2-44ac-b8a9-f6a10d294883)



