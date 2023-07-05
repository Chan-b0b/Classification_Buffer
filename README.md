# Classification_Buffer

Download Cifar10 file from https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders?resource=download

<img src="images/basic_architecture.png" alt="basic_architecture" width="400">

1. Random Sampling:
   - The model randomly samples data from a dataset.

2. Classification Model Scoring:
   - The sampled data is fed into a classification model.
   - The model assigns scores to the samples using cross-entropy values.
   - Higher scores indicate higher perceived difficulty.

3. Buffering:
   - The samples, along with their scores, are stored in a buffer.

4. Sampling from Buffer based on Score:
   - Samples are selected from the buffer for training.
   - The selection is biased towards samples with higher scores.
   - Higher-scored samples have a higher likelihood of being chosen for training.

5. Training with Sampled Data:
   - The selected samples are used to train the model.
   - This enables the model to focus on challenging samples based on their scores.

6. Iterative Process:
   - Steps 1 to 5 are repeated iteratively to continually update and improve the model's performance.
