from datasets import load_dataset
from dataset_helper import encode, to_tf_dataset
import numpy as np
from dataset_helper import create_dataset
import os
import json
import datetime
import numpy as np
import tensorflow as tf
from scheduler import CustomSchedule
from model import SnapthatT5

MY_METRICS = {
    'accuracy': tf.keras.metrics.SparseTopKCategoricalAccuracy(name='accuracy')
}

MY_OPTIMIZERS = {
    'Adam': tf.keras.optimizers.Adam
}

MY_MODELS = {
    'SnapthatT5': SnapthatT5
}

FIXED_HYPERPARAMS = {
    'model_arch': 'SnapthatT5', # needs to be in MY_MODELS
    'optimizer': 'Adam'         # needs to be in MY_OPTIMIZERS
}

class TFMixin:

    def get_data(self, name, pct, batch_size, num_workers):
        train_dataset = load_dataset(name, split=f'train[:{pct}%]')
        valid_dataset = load_dataset(name, split=f'validation[:{pct}%]')
        train_ds = train_dataset.map(encode)
        valid_ds = valid_dataset.map(encode)
        tf_train_ds = to_tf_dataset(train_ds)
        tf_valid_ds = to_tf_dataset(valid_ds)
        train_steps = int(np.ceil(len(train_dataset)/batch_size))
        valid_steps = int(np.ceil(len(valid_dataset)/batch_size))
        global_batch_size = num_workers * batch_size
        tf_train_ds = create_dataset(tf_train_ds, batch_size=global_batch_size, shuffling=True)
        tf_valid_ds = create_dataset(tf_valid_ds, batch_size=global_batch_size, shuffling=False)
        return tf_train_ds, tf_valid_ds, train_steps, valid_steps

    def configure_callbacks(self, tensorboard_logdir, checkpoint_savedir):
        tensorboard_logpath = tensorboard_logdir + "/" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logpath)
        checkpoint_filepath = checkpoint_savedir + "/T5-{epoch:04d}-{val_loss:.4f}.ckpt"
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            mode='min',
            save_best_only=True,
        )
        callbacks = [tensorboard_callback, checkpoint_callback]
        return callbacks

    def compile_model(
        self, 
        model_src="t5-small", 
        metrics=['accuracy'], 
        hyperparams={'learning_rate': 1e-6}
    ):
        hyperparams |= FIXED_HYPERPARAMS
        metrics = [MY_METRICS[m] for m in metrics]
        optimizer = MY_OPTIMIZERS[hyperparams['optimizer']](hyperparams['learning_rate'])
        model = MY_MODELS[hyperparams['model_arch']].from_pretrained(model_src)
        model.compile(optimizer=optimizer, metrics=metrics)
        return model