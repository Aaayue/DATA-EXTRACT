import json
import os
import sys
import glob
import math
from osgeo import gdalnumeric
from PIL import Image
import numpy as np
from os.path import join
from baikal.general.Vividict import Vividict
from shapely.ops import cascaded_union
from shapely.geometry import mapping
from shapely.geometry import shape
import fiona


def image_to_array(i: Image) -> np.ndarray:
    """
    Converts a Python Imaging Library array to a
    gdalnumeric image.
    """
    a = gdalnumeric.fromstring(i.tobytes(), dtype=np.int32)
    a.shape = i.im.size[1], i.im.size[0]
    return a


def image_to_array_byte(i: Image) -> np.ndarray:
    """
    Converts a Python Imaging Library array to a
    gdalnumeric image.
    """
    a = gdalnumeric.fromstring(i.tobytes(), dtype=np.int8)
    a.shape = i.im.size[1], i.im.size[0]
    return a


def array_to_image(a: np.ndarray) -> Image:
    """
    Converts a gdalnumeric array to a
    Python Imaging Library Image.
    """
    i = Image.frombytes(
        "L", (a.shape[1], a.shape[0]), (a.astype(np.int32)).tobytes())
    return i


def array_to_image_byte(a: np.ndarray) -> Image:
    """
    Converts a gdalnumeric array to a
    Python Imaging Library Image.
    """
    i = Image.frombytes(
        "I", (a.shape[1], a.shape[0]), (a.astype(np.int32)).tobytes())
    return i


def world_to_pixel(geo_matrix: tuple, x: int, y: int) -> tuple:
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate
    """
    up_left_x = geo_matrix[0]
    up_left_y = geo_matrix[3]
    x_dist = geo_matrix[1]
    pixel = int((x - up_left_x) / x_dist)
    line = int((up_left_y - y) / x_dist)
    return pixel, line


def pixels_to_world(geo_matrix: tuple, pixel: int, line: int) -> tuple:
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the geo-location of a img pixel coordinate
    """
    up_left_x = geo_matrix[0]
    up_left_y = geo_matrix[3]
    x_dist = geo_matrix[1]
    x = pixel * x_dist + up_left_x
    y = up_left_y - line * x_dist
    return x, y


def placelist_to_numeric(plist: list, state_id: dict, state_id_group: dict) -> list:
    """
    Function:
        turn input string list to numeric list
    Input:
        plist: string list
        state_id: dictionary from string to numeric
                like this:
                {"pulau":1
                 "kedah":2,...}
        state_id_group: dictionary from minus digit(group) to positive
                like this:
                {-1:[2,3,4]
                 -2:[4,5,6],...}
    Output:
        an int list.
    """
    nlist = []
    for p in plist:
        if p in state_id.keys():
            if state_id[p] > 0:  # if it is a state
                nlist.append(state_id[p])
            else:  # if it is a group
                nlist.extend(state_id_group[state_id[p]])
        else:
            raise Exception(
                "placelist_to_numeric(): key not found in state dictionary!"
            )
    # remove repeated elements
    olist = []
    for i in nlist:
        if i not in olist:
            olist.append(i)
    return olist


def get_reverse_dict(dic: dict) -> dict:
    """
    Function:
        reverse the dictionary
    Input:
        dic: dictionary to be reversed
    Output:
        rdic: reversed dict, keys are values in $dic,
                and values are keys in $dic.
    """
    rdic = {}
    for k, v in dic.items():
        rdic.setdefault(v, k)
    return rdic


def data_shuffle(data: np.ndarray) -> np.ndarray:
    """
    shuffle 2-D data along rows
    """
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    data = data[idx, :]
    return data


def data_shuffle_col(data: np.ndarray) -> np.ndarray:
    """
    shuffle 2-D data along colomns
    """
    row, col = data.shape
    idx = np.arange(col)
    np.random.shuffle(idx)
    data = data[:, idx]
    return data


def delete_nan_col(array: np.ndarray) -> np.ndarray:
    """
    Function:
        delete the colomns contains NaN values in given array $array
    return:
        a new array that deleted some colomns where contains NaN.
    """
    # row is feature, so process by each row
    row, col = array.shape
    arr = array
    for r in range(row):
        idx_nan = np.argwhere(np.isnan(arr[r, :]))
        idx_nan = idx_nan.flatten()
        if len(idx_nan):
            arr = np.delete(arr, idx_nan, axis=1)
    print(array.shape, "NaN deleted!, new shape is :", arr.shape)
    return arr


def delete_nan_row(array: np.ndarray) -> np.ndarray:
    """
    Function:
        delete the rows contains NaN and Inf values in given array $array
    return:
        a new array that deleted some rows where contains NaN.
    """
    # row is feature, so process by each row
    row, col = array.shape
    arr = array
    for r in range(col):
        idx_nan = np.argwhere(np.isnan(arr[:, r]))
        idx_nan = idx_nan.flatten()
        if len(idx_nan):
            arr = np.delete(arr, idx_nan, axis=0)

    row, col = arr.shape
    for r in range(col):
        idx_nan = np.argwhere(np.isinf(arr[:, r]))
        idx_nan = idx_nan.flatten()
        if len(idx_nan):
            arr = np.delete(arr, idx_nan, axis=0)

    row, col = arr.shape
    for r in range(col):
        idx_nan = np.argwhere(np.isneginf(arr[:, r]))
        idx_nan = idx_nan.flatten()
        if len(idx_nan):
            arr = np.delete(arr, idx_nan, axis=0)
    print(array.shape, "NaN, Inf, -Inf deleted!, new shape is :", arr.shape)
    return arr


def delete_999_row(array: np.ndarray) -> np.ndarray:
    """
    Function:
        delete the rows contains NaN values in given array $array
    return:
        a new array that deleted some rows where contains NaN.
    """
    # row is feature, so process by each row
    row, col = array.shape
    arr = array
    for r in range(col):
        idx_nan = np.where(arr == -999)
        # idx_nan = idx_nan.flatten()
        if len(idx_nan):
            arr = np.delete(arr, idx_nan, axis=0)
    print(array.shape, "-999 deleted!, new shape is :", arr.shape)
    return arr


def delete_0s_row(array: np.ndarray) -> np.ndarray:
    """
    Function:
        delete the rows contains NaN values in given array $array
    return:
        a new array that deleted some rows where contains NaN.
    """
    # row is feature, so process by each row
    row, col = array.shape
    arr = array
    for r in range(col - 1):
        idx_nan = np.where(arr <= 0)
        # idx_nan = idx_nan.flatten()
        if len(idx_nan):
            arr = np.delete(arr, idx_nan, axis=0)
    print(array.shape, "0s deleted!, new shape is :", arr.shape)
    return arr


def replace_invalid_value(array: np.ndarray, new_value: int) -> np.ndarray:
    """
    Function:
        replace the  NaN, Inf, -Inf values in given array $array
    return:
        a new array without NaN.
    """
    where_are_nan = np.isnan(array)
    array[where_are_nan] = new_value

    where_are_inf = np.isinf(array)
    array[where_are_inf] = new_value

    where_are_isneginf = np.isneginf(array)
    array[where_are_isneginf] = new_value
    return array


def set_vvdic_key_order(vv: Vividict) -> tuple:
    """
    set feature read order in 2-D list for a 2-D dict
    """
    ro_list = []
    time_list = []
    for k1 in vv.keys():
        k2s = list(vv[k1].keys())
        k2s.sort()
        for k2 in k2s:
            n = len(vv[k1][k2])
            ro_list.append([k1, k2, n])
            time_list.append(k2)  # just the 2nd layer keys: time
    return ro_list, time_list


def get_traindata_key_dict(ro_list: list) -> dict:
    """
    get a dict which feature name is key, colomn number is value
    """
    traindata_key_dic = {}
    for ro in ro_list:
        # ro is like this ["sentinel", "VV", 0]
        #                   ^ this is sensor name
        #                               ^ this is feature name
        #                                    ^ this is read/colomn order
        traindata_key_dic[ro[1]] = int(ro[2])
    return traindata_key_dic


def transform_inner_to_vvdic(dic: dict) -> Vividict:  # , dic:dict
    """
    transform a dict to vividict all layers
    """
    dicv = dic
    for key, value in dicv.items():  # dic.items():
        if type(value) is dict:
            dicv[key] = transform_inner_to_vvdic(value)
    return Vividict(dicv)


def save_json(json_file_path, file_dict):
    with open(json_file_path, "w") as fp:
        json.dump(file_dict, fp, ensure_ascii=True, indent=2)


def load_json(json_file_path):
    with open(json_file_path, "r") as fp:
        tmp = json.load(fp)
    return tmp


def add_root_path(img_pro_dict: dict, shp_reproj_dict: dict) -> (dict, dict):
    """
    Function:
        path add root for path, such as "data_pool/test" to "/home/tq/data_pool/test"
    """
    home_dir = os.path.expanduser("~")
    img_pro_dict = {
        key: {k: os.path.join(home_dir, v)
              for k, v in img_pro_dict[key].items()}
        for key in img_pro_dict.keys()
    }
    shp_reproj_dict = {k: os.path.join(home_dir, v)
                       for k, v in shp_reproj_dict.items()}
    # add home dir to each file path
    return img_pro_dict, shp_reproj_dict


def get_bands_into_a_dict(process_dict: dict, expression: str):
    img_path_dict = process_dict["ori_ras_path"]
    process_time = [int(d) for d in process_dict['year_date']]
    img_pro_dict = Vividict()
    for img_name in img_path_dict.keys():
        img_path = img_path_dict[img_name]
        pack_list = os.listdir(img_path)
        # folder only
        pack_list = [pack_name for pack_name in pack_list if os.path.isdir(join(
            img_path, pack_name))]
        pack_list.sort()
        for pack in pack_list:
            pack_path = join(img_path, pack)
            file_list = glob.glob(join(pack_path, expression))
            file_list.sort()
            if img_name is 'DEM':
                time_str = 'dem_time'
                img_pro_dict[img_name].setdefault(time_str, file_list)
            elif img_name is 'Sentinel_1':
                time_str = pack.split('_')[-1]
                year = int(time_str[:4])
                date = int(time_str[4:])
                if year == process_time[0] and process_time[1] <= date <= process_time[2]:
                    img_pro_dict[img_name].setdefault(time_str, file_list)
                else:
                    continue
            elif img_name is 'Landsat_8':
                time_str = pack
                if process_time[1] <= int(time_str) <= process_time[2]:
                    deep_pack = os.listdir(join(img_path, pack))
                    for d_pack in deep_pack:
                        new_time_str = str(
                            process_time[0]) + time_str + '-' + d_pack
                        # print(join(img_path, time_str, d_pack))
                        band_list = glob.glob(
                            join(img_path, time_str, d_pack, '*.tif'))

                        band_list = [k for k in band_list if 'band' in k]
                        file_list = [i for i in band_list if 'band1' not in i]
                        file_list.sort()
                        # print("file_list", file_list)
                        img_pro_dict[img_name].setdefault(
                            new_time_str, file_list)
                else:
                    continue
    return img_pro_dict


def get_bands_into_a_list(img_path: str, expression: str):
    filelist = glob.glob(img_path + expression)
    filelist.sort()
    return filelist


def print_progress_bar(now_pos: int, total_pos: int):
    n_sharp = math.floor(50 * now_pos / total_pos)
    n_space = 50 - n_sharp
    sys.stdout.write(
        "  ["
        + "#" * n_sharp
        + " " * n_space
        + "]"
        + "{:.2%}\r".format(now_pos / total_pos)
    )


def combine_npys(npy_list: list):
    n_files = len(npy_list)
    all_npy = None
    nn = 0
    for nf in npy_list:
        tmp_npy = np.load(nf)
        nn += 1
        print("{}/{} npy added".format(nn, n_files))
        if all_npy is None:
            all_npy = tmp_npy
        else:
            all_npy = np.vstack((all_npy, tmp_npy))
    return all_npy


def get_pathrow_data_label(s: str):
    sb = s.split("/")[-2]
    sl = sb.split("_")
    namestr = sl[2][0:3] + "-" + sl[2][3:6] + "-" + sl[3]
    return namestr


def get_landsat_by_pathrow(
        year: int, path: int, row: int, sensor: str, *, timerange="*"
):
    #

    search_str_base = "/home/tq/tq-data0ZZZ/landsat_sr/SSS/01/PPP/RRR/"

    pathstr = str(path).zfill(3)
    rowstr = str(row).zfill(3)
    yearstr = str(year)
    search_str_base = search_str_base.replace("PPP", pathstr)
    search_str_base = search_str_base.replace("RRR", rowstr)
    search_str_base = search_str_base.replace("SSS", sensor)
    # set start day end day
    if timerange == "*":
        daystart = 101
        dayend = 1231
    else:
        timerange_strs = timerange.split("-")
        if timerange_strs[0] != "*":
            daystart = int(timerange_strs[0])
        else:
            daystart = 101
        if timerange_strs[1] != "*":
            dayend = int(timerange_strs[1])
        else:
            dayend = 1231
    llist = []

    for tqn in range(5):
        tqnstr = str(tqn + 1)

        search_str = search_str_base.replace("ZZZ", tqnstr)
        print("searching " + search_str)

        # begin search dir
        if os.path.exists(search_str):
            dir_list = os.listdir(search_str)
            for cur_file in dir_list:
                # get fullpath
                namepart = cur_file.split("_")
                datestr = namepart[3]
                yearstr_file = datestr[0:4]
                monday_str = datestr[4:8]
                monday = int(monday_str)
                if daystart < monday < dayend:
                    pass
                else:
                    continue
                if yearstr_file != yearstr:
                    continue
                l8path = os.path.join(search_str, cur_file)
                if not l8path.endswith("/"):
                    l8path = l8path + "/"
                llist.append(l8path)
            # llist.extend(dir_list)
        else:
            continue

    return llist


def PR_boundary_intersect(image_PR_path, field_name, value_list, boundary_shape_path):
    """
    Function:
        generate intersect result based on ROI boundary and image PR path
    Input:
        image_PR_path: string, image PR shapefile path
        field_name: string, values of field in PR shapefile
        value_list: list (string), list of field values
        boundary_shape_path: string, ROI boundary shapefile path
    Output:
        intersect_shape: shape of intersect result
    """
    # read image PR shapefile
    multi_PRs = [shape(pol['geometry']) for pol in fiona.open(
        image_PR_path) if pol['properties'][field_name] in value_list]
    PR_shape = cascaded_union(multi_PRs)

    # get intersect shape
    boundary_shape = cascaded_union([shape(pol['geometry'])
                                     for pol in fiona.open(boundary_shape_path)])
    intersect_shape = boundary_shape.intersection(PR_shape)
    return intersect_shape


def save_shape_file(sour_shp_path, res_shape, res_shp_path):
    """
    Function:
        save shape object from shapely to shapefile
    Input:
        sour_shp_path: string, path of source shapefile which provide geoinfo
        res_shape: shape, shape object to save
        res_shp_path； string, path of result shapefile
    Output:
        None
    """
    with fiona.open(sour_shp_path, 'r') as source:
        with fiona.open(res_shp_path, 'w', **source.meta) as res_shp:
            res_poly = {'geometry': mapping(res_shape),
                        'properties': source.schema['properties'], }
            res_shp.write(res_poly)
