from metaflow import (
    FlowSpec,
    Parameter,
    step,
    batch,
    conda_base,
    tensorflow_parallel,
)

# from dotenv import load_dotenv
# try:
#     load_dotenv(verbose=True, dotenv_path="my_env.env")
# except:
#     print("No dot env!")

# @conda_base(
#     libraries={
#         "dill": "0.3.5.1",
#         "datasets": "2.13.0",
#         "transformers": "4.29.2",
#         "tensorflow": "2.11.1",
#         "cudatoolkit": "11.8.0",
#         "sentencepiece": "0.1.99",
#     },
#     python="3.9.16",
# )

class T5TensorFlowFlow(FlowSpec):
    num_parallel = Parameter(
        name="num_parallel",
        default=2,
    )

    batch_size = Parameter(
        name="batch_size",
        default=4,
    )

    epochs = Parameter(
        name="epochs",
        default=1,
    )

    tensorboard_logdir = Parameter(
        name="tensorboard_logdir", 
        default="./data/experiments/t5/logs",
    )

    checkpoint_savedir = Parameter(
        name="checkpoint_savedir",
        default="./data/experiments/t5/models",
    )

    @step
    def start(self):
        print("start")
        self.next(self.train)

    # @step
#     def load_data(self):
#         print("load data")
#         from datasets import load_dataset
#         from dataset_helper import encode, to_tf_dataset

#         train_dataset = load_dataset('squad', split='train[:10%]')
#         # valid_dataset = load_dataset('squad', split='validation')
#         print("load done")
#         print("dataset feature: ", train_dataset.features)

#         train_ds = train_dataset.map(encode)
#         # valid_ds = valid_dataset.map(encode)
#         # ex = next(iter(train_ds))
#         # print("Example data from the mapped dataset: \n")
#         # for e in ex:
#         #     print(e, type(ex[e]), ex[e])

#         print("encode done")

#         self.train_ds = train_ds

        # self.tf_train_ds = to_tf_dataset(train_ds)
        # self.tf_valid_ds = to_tf_dataset(valid_ds)
        # print("to_tf_dataset done")

        # self.tf_train_ds = create_dataset(tf_train_ds, batch_size=self.batch_size, shuffling=True)
        # self.tf_valid_ds = create_dataset(tf_valid_ds, batch_size=self.batch_size, shuffling=False)

    
    # @batch(cpu=4, gpu=3, queue="metaflow-gpu-fusmltrn")
    # @tensorflow_parallel
    @step
    def train(self):
        import os
        import json
        import datetime
        import numpy as np
        import tensorflow as tf
        from scheduler import CustomSchedule
        from model import SnapthatT5
        from dataset_helper import create_dataset
        
        from datasets import load_dataset
        from dataset_helper import encode, to_tf_dataset
        
        train_dataset = load_dataset('squad', split='train[:10%]')
        valid_dataset = load_dataset('squad', split='validation[:10%]')
        print("load done")
        print("dataset feature: ", train_dataset.features)

        train_ds = train_dataset.map(encode)
        valid_ds = valid_dataset.map(encode)
        self.tf_train_ds = to_tf_dataset(train_ds)
        self.tf_valid_ds = to_tf_dataset(valid_ds)
        print("to_tf_dataset done")

        # tf_config = json.loads(os.environ["TF_CONFIG"])
        # num_workers = len(tf_config["cluster"]["worker"])
        num_workers = 1
        global_batch_size = num_workers * self.batch_size
        # strategy = tf.distribute.MultiWorkerMirroredStrategy()

        train_steps = int(np.ceil(len(train_dataset)/self.batch_size))
        valid_steps = int(np.ceil(len(valid_dataset)/self.batch_size))

        tf_train_ds = create_dataset(self.tf_train_ds, batch_size=global_batch_size, shuffling=True)
        tf_valid_ds = create_dataset(self.tf_valid_ds, batch_size=global_batch_size, shuffling=False)

        tensorboard_logpath = self.tensorboard_logdir + "/" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logpath)

        checkpoint_filepath = self.checkpoint_savedir + "/T5-{epoch:04d}-{val_loss:.4f}.ckpt"
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            mode='min',
            save_best_only=True,
        )

        callbacks = [tensorboard_callback, checkpoint_callback]
        metrics = [tf.keras.metrics.SparseTopKCategoricalAccuracy(name='accuracy')]

        # learning_rate = CustomSchedule()
        learning_rate = 1e-6

        # with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        model = SnapthatT5.from_pretrained("t5-small")
        model.compile(optimizer=optimizer, metrics=metrics)

        print("begin training.")
        model.fit(
            tf_train_ds,
            epochs=self.epochs,
            steps_per_epoch=train_steps,
            validation_data=tf_valid_ds, 
            validation_steps=valid_steps,
            callbacks=callbacks, 
        )

        print("training done.")

        self.next(self.end)

    # @step
    # def multinode(self, inputs):
    #     self.next(self.end)
    
    @step
    def end(self):
        pass
    

if __name__ == "__main__":
    T5TensorFlowFlow()
