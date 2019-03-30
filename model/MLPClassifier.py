import numpy as np
import os
import time
import logging
from sklearn.metrics import f1_score
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier


class MLPNetWork():

    logging.basicConfig(level=logging.DEBUG)
    my_logger = logging.getLogger(__name__)

    def __init__(
        self,
        data_file: np.array,
        model_folder: str,
        *,
        label_str: str = "",
        crossValidation_num: int = 3,
        hidden_layer_sizes: tuple = (512, 256, 128),
        learning_rate_init: float = 0.001,
        max_iter: int = 500,
        test_size: float = 0.3,
    ) -> str:
        # initialize parameters
        self.data_file = data_file
        self.model_folder = model_folder
        self.crossValidation_num = crossValidation_num
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.test_size = test_size
        self.label_str = label_str

    def model(self):
        # split train feature and label
        data = np.load(self.data_file)
        train_feature = data['features']
        train_label = data['labels']

        print("feature shape: ", train_feature.shape)
        print("label shape: ", train_label.shape)

        # save model
        model_name = (
            "MLPClassifier"
            + "_"
            + self.label_str
            + time.strftime("%Y%m%d%H%M%S", time.localtime())
            + ".m"
        )
        model_path = os.path.join(self.model_folder, model_name)
        model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
        )

        (train_sub_feature,
         test_sub_feature,
         train_sub_label,
         test_sub_label) = train_test_split(
            train_feature, train_label, test_size=self.test_size
        )

        model = model.fit(train_sub_feature, train_sub_label)
        print(
            "Training accuracy: %f"
            % (model.score(train_sub_feature, train_sub_label))
        )

        print(
            "Testing accuracy: %f" % (model.score(
                test_sub_feature, test_sub_label))
        )
        conf_mat = confusion_matrix(
            test_sub_label, model.predict(test_sub_feature)
        )
        f1 = f1_score(test_sub_label, model.predict(test_sub_feature))

        print(
            "Confusion matrix:\n",
            conf_mat
        )

        print(
            "F1 score: %f" % (f1)
        )

        try:
            joblib.dump(model, model_path)
            self.my_logger.info(
                "Decision tree model saved! Result path: %s", model_path)

        except Exception as e:
            self.my_logger.error(
                "{}, Save decision tree model failed!".format(e))
            return None

        return model_path


if __name__ == '__main__':
    file = '/home/zy/data2/citrus/demo/sample_result/125040_noDEM_20190307T204448/TD_S3_L3a_20190307T204448_TRAIN.npz'
    MLP = MLPNetWork(file, "/home/zy/data_pool/U-TMP/TMP")
    MLP.model()
