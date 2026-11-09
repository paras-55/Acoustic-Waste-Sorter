Sound-Based Waste Classification
==================================

1_ESP32_Code/
  Arduino sketches for ESP32.

2_Audio_Dataset/Plastic/
  Labelled plastic impact recordings (.wav).

2_Audio_Dataset/Metal/
  Labelled metal impact recordings (.wav).

3_ML_Model/
  Python training scripts and trained model files.

4_Test_Results/
  Test logs, charts, and evaluation reports.

Suggested audio filenames:
  plastic_object01_trial01.wav
  metal_object01_trial01.wav

Keep the impact surface, drop height, and microphone position consistent.
For reliable evaluation, keep recordings from each physical object in only
one of the training or test sets.
