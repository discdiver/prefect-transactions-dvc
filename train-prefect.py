
import numpy as np
import os
from prefect import flow, task
from prefect.transactions import get_transaction, transaction
import subprocess
from shlex import split
import sys
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras import applications
from tensorflow.keras.callbacks import CSVLogger
from tqdm.keras import TqdmCallback


@task(retries=3)
def add_data(dataset_name: str = "data"):
    """Fetch and add data for model training"""
    subprocess.run(
        split(
            f"dvc get https://github.com/iterative/dataset-registry \
            tutorials/versioning/{dataset_name}.zip"
        )
    )
    subprocess.run(split(f"unzip -q {dataset_name}.zip"))
    subprocess.run(split(f"rm -f {dataset_name}.zip"))
    subprocess.run(split(f"dvc add {dataset_name}"))


@task(log_prints=True)
def train_model():
    """Train model for image classification"""

    pathname = os.path.dirname(sys.argv[0])
    path = os.path.abspath(pathname)

    # dimensions of our images.
    img_width, img_height = 150, 150

    top_model_weights_path = "model.weights.h5"
    train_data_dir = os.path.join("data", "train")
    validation_data_dir = os.path.join("data", "validation")
    cats_train_path = os.path.join(path, train_data_dir, "cats")
    nb_train_samples = 2 * len(
        [
            name
            for name in os.listdir(cats_train_path)
            if os.path.isfile(os.path.join(cats_train_path, name))
        ]
    )
    nb_validation_samples = 800
    epochs = 10
    batch_size = 10

    datagen = ImageDataGenerator(rescale=1.0 / 255)

    # train bottleneck features
    model = applications.VGG16(include_top=False, weights="imagenet")

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False,
    )
    bottleneck_features_train = model.predict(generator, nb_train_samples // batch_size)
    np.save(open("bottleneck_features_train.npy", "wb"), bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False,
    )
    bottleneck_features_validation = model.predict(
        generator, nb_validation_samples // batch_size
    )
    np.save(
        open("bottleneck_features_validation.npy", "wb"), bottleneck_features_validation
    )

    # Train the top of the model based on our image data
    train_data = np.load(open("bottleneck_features_train.npy", "rb"))
    train_labels = np.array(
        [0] * (int(nb_train_samples / 2)) + [1] * (int(nb_train_samples / 2))
    )

    validation_data = np.load(open("bottleneck_features_validation.npy", "rb"))
    validation_labels = np.array(
        [0] * (int(nb_validation_samples / 2)) + [1] * (int(nb_validation_samples / 2))
    )

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

    print("Training model")

    history = model.fit(
        train_data,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(validation_data, validation_labels),
        verbose=0,
        callbacks=[TqdmCallback(), CSVLogger("metrics.csv")],
    )
    model.save_weights(top_model_weights_path)
    return history


@task
def git_track(tag: str = "v1.0", img_count: int = 1000):
    """Track model changes in git"""
    subprocess.run(split("dvc add model.weights.h5"))
    subprocess.run(
        split(f"git add data.dvc model.weights.h5.dvc metrics.csv .gitignore")
    )
    subprocess.run(
        split(f"git commit -m '{tag} model, trained with {img_count} images'")
    )
    subprocess.run(split(f"git tag -a '{tag}' -m 'model {tag}, {img_count} images'"))


@task
def check_model_val(history):
    """Check accuracy score of validation set"""
    if history.history["val_accuracy"][-1] < 0.95:
        raise ValueError(f"Validation accuracy is too low:{history.history['val_accuracy'][-1]}")


@git_track.on_rollback
def rollback_workspace(transaction):
    """Automatically roll back the workspace to the previous commit if model evaluation fails"""
    subprocess.run(split("git checkout HEAD~1"))
    subprocess.run(split("dvc checkout"))
    subprocess.run(split(f"git tag -d {transaction.get('tagging')}"))
    print(f"Rolling back workspace from {transaction.get("tagging")} to previous commit because validation accuracy was too low")

 
@flow
def pipeline(dataset_name: str, tag: str, img_count: int, initial_run: bool = False):
    """Pipeline for training model and checking validation accuracy"""
    add_data(dataset_name=dataset_name)
    with transaction() as txn:
        history = train_model()
        git_track(tag=tag, img_count=img_count)
        txn.set("tagging", tag)
        if not initial_run:
            check_model_val(history=history)


if __name__ == "__main__":
    dataset_info = [["data", "v1.0", 1000], ["new-labels", "v2.0", 2000]]
    pipeline(*dataset_info[0], initial_run=True)
    pipeline(*dataset_info[1])
