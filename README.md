# Use Prefect 3.0's new transaction capabilities to train machine learning models more effectively

This repository accompanies an article that shows how to use Prefect 3's transaction capabilities with DVC for resilient data and ML pipelines.
TK

This example is adapted from DVC's [Data and Model Versioning tutorial](https://dvc.org/doc/tutorials/versioning), which in turn is based on "Building powerful image classification models using very little data" from blog.keras.io.

From the DVC tutorial script:

In our example we will be using data that can be downloaded at:
<https://www.kaggle.com/tongpython/cat-and-dog>

In our setup, it expects:

- a data/ folder
- train/ and validation/ subfolders inside data/
- cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-X in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 0-X in data/train/dogs
- put the dog pictures index 1000-1400 in data/validation/dogs

We have X training examples for each class, and 400 validation examples
for each class. In summary, this is our directory structure:

```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
