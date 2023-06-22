from metaflow import (
    FlowSpec,
    Parameter,
    step,
    batch,
    conda_base,
    tensorflow_parallel
)

from tf_utils import TFMixin

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
class T5TensorFlowFlow(FlowSpec, TFMixin):

    num_parallel = Parameter(
        name="num_parallel",
        default=2,
        type=int
    )

    batch_size = Parameter(
        name="batch_size",
        default=4,
        type=int
    )

    epochs = Parameter(
        name="epochs",
        default=1,
        type=int
    )

    tensorboard_logdir = Parameter(
        name="tensorboard_logdir", 
        default="./data/experiments/t5/logs",
        type=str
    )

    checkpoint_savedir = Parameter(
        name="checkpoint_savedir",
        default="./data/experiments/t5/models",
        type=str
    )

    model_arch = Parameter(
        name="model_arch",
        default="SnapthatT5", # add/view options in utils.py
        type=str
    )

    model_src = Parameter(
        name = "model_src",
        default = "t5-small",
        type=str
    )

    learning_rate = Parameter(
        name = "learning_rate",
        default=1e-6,
        type=float
    )

    @step
    def start(self):
        print("start")
        self.next(self.train)

    # @batch(cpu=4, gpu=3, queue="metaflow-gpu-fusmltrn")
    # @tensorflow_parallel
    @step
    def train(self):
        
        tf_train_ds, tf_valid_ds, train_steps, valid_steps = self.get_data(
            'squad', pct='10', batch_size = self.batch_size, num_workers=1)

        callbacks = self.configure_callbacks(self.tensorboard_logdir, self.checkpoint_savedir)

        self.tuning_set = {'learning_rate': self.learning_rate}
        model = self.compile_model(model_src = self.model_src, hyperparams=self.tuning_set)

        print("begin training.")
        model.fit(
            tf_train_ds,
            epochs=self.epochs,
            steps_per_epoch=train_steps,
            validation_data=tf_valid_ds, 
            validation_steps=valid_steps,
            callbacks=callbacks, 
            verbose=2
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
