# Use Prefect 3's new transaction capabilities to train machine learning models more effectively

An example with Prefect and DVC for improved versioning of datasets and ML models

This repo accompanies an article that shows how to use Prefect 3's transaction capabilities to roll back task's side effects when model performance doesn't meet your specifications.

This example is adapted from DVC's Model Versioning tutorial. The original tutorial is available [here](https://dvc.org/doc/tutorials/versioning).

The tutorial has been modified so that all commands after the setup below are included in the train-prefect.py script.

## Setup

We recommend using a Python virtual environment to run this example. To create a virtual environment with venv and install the required packages, run the following commands:

```bash
python -m venv .env   
source .env/bin/activate
pip install -r requirements.txt
```

This example assumes you have a Prefect 3 server instance running or have a Prefect Cloud account with your CLI authenticated. See the [Quickstart instructions](https://docs-3.prefect.io/3.0/get-started/quickstart#connect-to-a-prefect-api), if needed.

## Run the Python script

```bash
python train-prefect.py
```

The script downloads 1,000 cat and dog images and uses them to fine-tune a pre-trained VGG16 model using Tensorflow Keras.
This is our initial model.

We use git and DVC to version the model and the dataset.
DVC is especially handy when working with large datasets and models that don't fit into git repositories easily.

We then add more training data and retrain our model.
We track the model and data using DVC and git.

One powerful aspect of Prefect's transactions is the ability to create a rollback hook to undo side effects if a task fails.

In this example, we have a flow-decorated function named `pipeline` that calls our tasks.
Our pipeline includes a `with transaction` context block that calls a task that checks our model's performance on a validation dataset.
If the model's validation accuracy doesn't reach 95% we raise an error.
This error means that our flow run  undo saving the data and model by rolling back our workspace if the previous model and data.
This is a bit of a contrived example, but it shows the power of Prefect's transaction semantics.

Note that we are able to pass information for use even in the rollback function by using transaction's `get` and `set` methods.

As we've seen, Prefect 3's transaction capabilities allow us to easily discard the data and model weights if a model's performance doesn't meet our specifications. We could have even undone code changes in other training scripts if we had used them to experiment with our model performance.

## Caching

Prefect 3's caching is based on transactions. TK
Any tasks that we include in a `with transaction` context block will not be cached by Prefect if an error is raised.
All the tasks in the block will run together, or not run together if they are cached.

See the [task caching docs](https://docs.prefect.io/latest/develop/task-caching) for more details.

In this example, we use a transaction to group the `committing` and `model quality check`. If the model quality check fails, the `on_rollback` hook will reset the git and dvc commits, by

In our example, if an error

```python


Transactions go through the following steps:


TK consider whether want to show caching. Maybe show how we don't need to fetch data if something hasn't changed - maybe the dataset name? The thinking being that we always use different dataset names, not the same one at our org.
