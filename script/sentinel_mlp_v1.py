import os
import time
import socket
import pprint
import numpy as np
import pandas as pd
from common import logger
from baikal.model.RunPredictor import run_predictor
from baikal.model.LSTM import LSTModel
from baikal.general.common import (
    save_json,
    load_json,
    get_bands_into_a_list,
    combine_npys,
    get_bands_into_a_dict,
    set_vvdic_key_order,
)
from baikal.ground_truth.TrainDataExtractor import TrainDataExtractorV2
from baikal.ground_truth.TrainDataSG import TrainDataSG
from baikal.ground_truth.TrainDataFlat import batch_run
from os.path import join

# for model test
import glob
import json


def go_get_training(process_dict: dict) -> dict:
    """
    using label shape and tif to get training data
    """
    # get all need data for processing
    process_dict["img_pro_dict"] = get_bands_into_a_dict(
        process_dict, "*_prj_cliped.tif"
    )

    # get a read order list
    read_list, time_list = set_vvdic_key_order(
        process_dict["img_pro_dict"]
    )
    process_dict["read_order_list"] = read_list
    process_dict["time_order_list"] = time_list
    pprint.pprint(process_dict)
    # run get training data
    tde = TrainDataExtractorV2(
        process_dict,
        sample_label="S3_L3a_" + time.strftime("%Y%m%dT%H%M%S"),
        label_keep_list=[1],
        is_binarize=True,
    )
    traindata_dict, trainlab, process_dict[
        "traindata_path_npz"
    ] = tde.go_get_mask_2npy()
    sg_file = process_dict["traindata_path_npz"]
    print(sg_file)
    SG = TrainDataSG(
        file=sg_file,
        quantity=process_dict["chunk"],
        year=process_dict['year_date'][0],
        start_day=process_dict['year_date'][1],
        end_day=process_dict['year_date'][2]
    )
    SG.batch_run()
    sg_list = glob.glob(join(process_dict["work_path"], '*.npz'))
    sg_list = [s for s in sg_list if '_' +
               str(process_dict["chunk"]) + '_' in s]
    pprint.pprint(sg_list)
    process_dict["traindata_path_npz"] = batch_run(sg_list)
    # save process dict
    result_file = os.path.join(
        process_dict["work_path"],
        "traindata_" + time.strftime("%Y%m%dT%H%M%S") + ".json",
    )
    save_json(result_file, process_dict)
    print("Extract data finish!")
    return process_dict


def go_to_training(process_dict: dict) -> dict:
    """
    using training data to run model
    """
    # get training data
    train_file = process_dict["traindata_path_npz"]
    traindata = np.load(train_file)
    train_lab = traindata['labels'].tolist()
    feature_length = traindata["features"].shape[1]
    print("total label: %d" % sum(train_lab))

    # TODO: get model save file from process dict
    # model_dir = os.path.join(process_dict["work_path"], process_dict["model_dir"])
    model_dir = os.path.join(process_dict["work_path"], 'lstm_bn_test2')
    timestep = pd.date_range(
        process_dict['year_date'][0] + process_dict['year_date'][1],
        process_dict['year_date'][0] + process_dict['year_date'][2],
        freq="5D")

    print(feature_length, len(timestep))
    LSTM = LSTModel(
        model_dir=model_dir,
        file=train_file,
        # TODO: calculate num_input based on SG param
        num_input=int(feature_length / len(timestep)),
        timesteps=len(timestep),
    )
    process_dict["model_path"] = LSTM.sess_run()

    # save process dict
    result_file = os.path.join(
        process_dict["work_path"],
        "model_" + time.strftime("%Y%m%dT%H%M%S") + ".json"
    )

    save_json(result_file, process_dict)
    print("Training model finish!")
    return process_dict


def go_to_predictor(process_dict: dict) -> dict:
    """
    using model ro run prodict
    """
    # run predictor
    status, process_dict["result_path"] = run_predictor(process_dict)
    if status:
        print("result file: ", process_dict["result_path"])
    else:
        print("Failed!")
    pprint.pprint(process_dict)

    # save process dict
    result_file = os.path.join(
        process_dict["work_path"], "product_" +
                                   time.strftime("%Y%m%dT%H%M%S") + ".json"
    )
    save_json(result_file, process_dict)
    print("Predict result finish!")
    return process_dict


def control_tool(json_file, process_dict, process_label=0) -> bool:
    """
    Function:
        0: get training data -> model training -> run prodict
        1: get training data -> model training
        2: get training data
        3: model training
            must be give the training process dict
        4: run prodict
            must be give the product
    """
    if process_label in [0, 1, 2]:
        process_dict_training = go_get_training(process_dict)
        if process_label in [0, 1]:
            process_dict_model = go_to_training(process_dict_training)
            if process_label == 0:
                process_dict_product = go_to_predictor(process_dict_model)
                print("product finished! %d" % process_label)
            else:
                print("product finished! %d" % process_label)
        else:
            print("product finished! %d" % process_label)
    elif process_label == 3:
        with open(json_file) as f:
            process_dict = json.load(f)
        process_dict_model = go_to_training(process_dict)
        print("model training finished! %d" % process_label)
    elif process_label == 4:
        process_dict_product = go_to_predictor(process_dict)
        print("product finished! %d" % process_label)
    else:
        print("please check the process label!")


if __name__ == "__main__":
    # for input
    # TODO: the DEM source must be contained in process_dict
    print('start')
    home_dir = os.path.expanduser('~')
    process_dict = {
        "ori_ras_path": {
            "DEM": join(home_dir, "data2/E-EX/Citrus/125041/DEM"),
            "Landsat_8": join(home_dir, "data2/E-EX/Citrus/125041/Landsat8"),
            "Sentinel_1": join(home_dir, "data2/E-EX/Citrus/125041/Sentinel1"),
        },
        "img_pro_dict": {},
        "pro_ras_list": [{
            "Sentinel_1": join(home_dir, "data2/E-EX/Citrus/125041_s1/"),
            "DEM": "/home/tq/data2/citrus/hunan_data/hunan_DEM/125040_125041/",
            "Landsat_8": "/home/tq/data2/citrus/hunan_data/hunan_L8/125040_125041/",
        }, ],
        "work_path": "",
        "model_path": "",
        "traindata_path_npz": "",
        "field_name": "label",
        "res_label": "hn_125_Demo_0227",
        # ===========================
        "model_dir": "lstm_bn",
        "chunk": 4000,
        "year_date": ['2018', '0101', '1231'],
        # ===========================
        "shp_reproj_dict": {
            "samples":
                join(home_dir,
                     # "data2/citrus/label/hunan/hunan_20190305.shp")
                     # "data2/citrus/label/hunan/hunan_20190312.shp")
                     "data2/citrus/label/hunan/hunan_citrus_label_20190314.shp")
        },
        "read_order_list": "",
    }

    """
    state = 
        0: get training data -> model training -> run prodict
        1: get training data -> model training
        2: get training data
        3: model training
    """
    state = 3
    source_path_list = glob.glob(
        join(
            home_dir,
            'tq-data04/TQLS/*/*/*/Landsat8fake',
            process_dict['year_date'][0]
        )
    )

    i = 0
    label_valid_tile = [
        "49RCN",
        "49RCM",
        "49RCL",
        "49RBL",
        "49RDP",
        "49RDN",
        "49RDM",
        "49RDL",
        "49RDK",
        "49REQ",
        "49REP",
        "49REN",
        "49REM",
        "49REL",
        "49REK",
        "49REJ",
        "49RFN",
        "49RFM",
        "49RFL",
        "49RFK",
        "49RFJ",
        "49RGN",
        "49RGM",
        "49RGL",
        "49RGK",
        "49RGJ",
    ]
    for path in source_path_list:
        i += 1
        if state in [0, 1, 2]:
            # make work path
            work_path = join(home_dir, "data2/citrus/demo/sample_result/MRGS/")
            tile = path.split('/')
            tile = tile[5] + tile[6] + tile[7]
            if tile not in label_valid_tile:
                continue
            else:
                print('PROCESS TILE ', tile)
                process_dict["work_path"] = "{}{}_{}_{}/".format(
                    work_path, socket.gethostname(),
                    tile+'_new',
                    time.strftime("%Y%m%dT%H%M%S"))
                if not os.path.exists(process_dict["work_path"]):
                    os.makedirs(process_dict["work_path"])
                local_path = path.split('Landsat8fake')[0]
                print(local_path)
                process_dict['ori_ras_path']['DEM'] = join(local_path, 'DEM')
                process_dict['ori_ras_path']['Landsat_8'] = path
                process_dict['ori_ras_path']['Sentinel_1'] = join(
                    local_path, 'Sentinel1')
                control_tool(None, process_dict, process_label=state)
        elif state == 3:
            j_file = '/home/tq/data2/citrus/demo/sample_result/125040_125041_DEM/traindata_20190311T165850.json'
            control_tool(j_file, None, process_label=state)
            # TODO: data faltten for extracted files from different tiles
            # TODO: control handler for production only
