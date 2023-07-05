# Classification_Buffer

Download Cifar10 file from https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders?resource=download

<img src="images/basic_architecture.png" alt="basic_architecture" width="400">

1. Randomly sample from the dataset.
2. Score the samples using a classification model, using metrics like crossentropy. Higher scores indicate higher difficulty.
3. Store the samples in a buffer.
4. Sample from the buffer based on the score value. Higher-scored samples have a higher likelihood of being selected.
5. Train the model using the sampled data.
6. Repeat steps 1-5.
