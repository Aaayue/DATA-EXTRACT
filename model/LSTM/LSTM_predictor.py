import numpy as np
import tensorflow as tf
import logging
import os


class TensorFlowPredictor:
    # use this as some trick for py 3.7 logger picklable
    # https://stackoverflow.com/questions/3375443/how-to-pickle-loggers
    logger = logging.getLogger(__qualname__)  # noqa: F821

    def __init__(self, model_path):
        self.tf_session = self.init_tf_session()

        saver = tf.train.import_meta_graph(model_path)
        model_dir = os.path.dirname(model_path)
        saver.restore(self.tf_session, tf.train.latest_checkpoint(model_dir))

    @classmethod
    def init_tf_session(cls):
        conf = tf.ConfigProto()
        conf.gpu_options.allow_growth = True
        tf_session = tf.Session(config=conf)
        return tf_session

    def predict_the_fuck(self, pixels_to_be_predict):
        """
        Args:
        ----
            input: pixels_to_be_predict = 3-d, [pixelNum, bandNum, timeSeriesNum]
        """

        self.logger.debug("pixels_to_be_predict shape %s",
                          pixels_to_be_predict.shape)

        pixels_to_be_predict = pixels_to_be_predict.transpose(0, 2, 1)

        prediction = self.tf_session.run(
            ["prediction:0"],
            feed_dict={"input_x:0": pixels_to_be_predict,
                       "dropout_prob:0": 0.0},
        )

        prediction = prediction[0]

        # zero will reamin as nodata, each origin arg index will increate by 1,
        # and last value means other will remap to 255

        (pixel_len, crop_len) = prediction.shape
        other_val = crop_len
        crop_type_index = np.argmax(prediction, axis=1) + 1
        crop_type_index[crop_type_index == other_val] = 255

        self.logger.debug("done ml_predict")
        crop_type_index = prediction
        return crop_type_index
