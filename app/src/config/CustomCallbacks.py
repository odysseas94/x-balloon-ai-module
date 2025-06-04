import datetime
import copy
import tensorflow as tf



class MetricsCallback(tf.keras.callbacks.Callback):
    metrics = {}
    counter = 0

    def on_epoch_begin(self, epoch, logs=None):
        print('Evaluating: epoch {} begin at {}'.format(epoch, datetime.datetime.now().time()), logs)

    def on_epoch_end(self, epoch, logs=None):
        self.counter += 1
        if logs:
            self.metrics[self.counter] = copy.copy(logs)
        self.load_from_server.write_down_json_metrics(self.metrics)


    def __init__(self, load_from_server):
        super().__init__()
        self.load_from_server = load_from_server
        self.metrics = {}
