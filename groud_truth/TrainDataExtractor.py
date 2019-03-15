import logging
import math
import os
import numpy as np
import glob
from os.path import join
import subprocess
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from functools import reduce
from baikal.general.Vividict import Vividict
from baikal.general.common import *
from baikal.general.calc_mask_by_shape import (
    calc_mask_by_shape,
    calc_mask_by_shape_block,
)


class TrainDataExtractorV2:
    my_logger = logging.getLogger(__qualname__)

    def __init__(
        self,
        process_dic: dict,
        *,
        label_keep_list: list = [1],
        is_binarize: bool = True,
        mask_value: int = 1,
        sample_label: str = "test",
        label: int = 1,
    ):
        """
        Function:
            initialize TrainDataExtractor class
        Input:
            raster_dic:a 2-layered dict contains the raster files
                       as the model input.
                       like this:
                        {"sensor-1":{"band-1":"band path",
                                     "band-2":"band path"],
                                    ...}
                         "sensor-2":{"band-1":"band path",
                                     "band-2":"band path",
                                    ...}
                        }
            sampe_shp_path: training shape file path, a file that contains
                        all the ground-truth polygons drawn by human.
            bound_path:  boundary shape file of entire research area, eg:
                        a country shp-file in state/province level.
            work_path:   a path that stores temporary files or results.
            label_dict:  a dictionary contains label of each ground-truth
                        polygon, a field named "label" is necessary.
            read_order_list: 2d list of read order
            aux_data:   a auxilary data dictionay, including band_list and its
                        invalid value info.

            *: optional parameters:
            field_name:  field name specified for mask making, "label" for
                        default.
            mask_value:  default mask value for mask-making, 1 for default;
            place_label: a string for final .npy file naming. means place
                        name.
            time_label:  a string for final .npy file naming. means img time.
            sample_label: a string for final .npy file naming. means the
                        ground-truth file label.
        """
        # set dictionary variables
        img_dict = process_dic["img_pro_dict"]
        ori_dict = process_dic["ori_ras_path"]
        shp_dict = process_dic["shp_reproj_dict"]
        # img_dict, shp_dict = add_root_path(img_dict, shp_dict)
        img_dict = transform_inner_to_vvdic(img_dict)   # 利用函数实现字典多层赋值操作

        self.img_dict = img_dict
        self.shp_dict = shp_dict
        self.ori_dict = ori_dict
        self.shp_label_path = shp_dict["samples"]
        self.work_path = process_dic["work_path"]

        # key para
        self.field_name = process_dic["field_name"]
        self.mask_value = mask_value
        self.label_keep_list = label_keep_list
        self.ref_coef = 10000
        self.isbinarize = is_binarize
        self.NDV = -999

        self.read_order_list = process_dic["read_order_list"]
        self.__n_block = 1

        # set joint char that connects each part in output filename
        self.__join_char = "_"

        # star symbols rotating while extracting traindata, pretending the
        # program is still alive...
        self.str_arrs = ["~", "/", "|", "\\"]

        self.sample_label = sample_label
        self.id = label
        self.feature = []
        self.label = []

    def set_join_char(self, char: str) -> bool:
        """
        set the join char if you don't like "_"
        """
        if len(char) != 1:
            self.my_logger.error("set_join_char(): char length not 1!")
            return False
        if char in ["\\", "|", "/", ":", "*", "?", '"', "<", ">", " "]:
            self.my_logger.error("set_join_char(): illegal char!")
            return False
        self.__joinchar = char
        return True

    def set_keep_label(self, klist: list) -> bool:
        self.label_keep_list = klist
        return True

    def set_NDV(self, ndv: int) -> bool:
        """
        set nodata value
        """
        self.NDV = ndv
        return True

    def set_block_num(self, num: int):
        """
        set blocking number
        """
        self.__n_block = num
        self.my_logger.info("block number has set to {}".format(num))

    def set_label_dicts(self, dic: dict):
        """
        Function:
            set self.__label_dict and __label_dict_R if you have a
            new label-dictionary.
        """
        self.__label_dict = dic
        self.__label_dict_R = get_reverse_dict(dic)
        self.my_logger.info("label dictionary updated!~")

    def check_proj(self, shape_path: str, ras_path: str) -> bool:
        """
        Function:
            check the projection of shapefile and raster,
            if inconsist, raise an exception.
        Input:
            shape_path: path of shapefile
            ras_path:   path of raster image
        Output:
            No output. just continue if no exception is thrown.
        """
        # open shapefile
        try:
            shapef = ogr.Open(shape_path)
            lyr = shapef.GetLayer(0)
            shp_spatial_ref = lyr.GetSpatialRef()
        except Exception as e:
            self.my_logger.error("open shape file error: %s", shape_path)
            return False

        # open rasterfile
        try:
            ds = gdal.Open(ras_path)
            ras_proj = ds.GetProjection()
            ras_spatial_ref = osr.SpatialReference(wkt=ras_proj)
        except Exception as e:
            self.my_logger.error("open raster file error: %s", ras_path)
            return False

        if shp_spatial_ref.GetAttrValue("projcs") != ras_spatial_ref.GetAttrValue(
            "projcs"
        ):
            self.my_logger.error(
                "check_proj(): incorrect projection, check again!~")
            return False
        else:
            self.my_logger.info("projection check passed!")
        return True

    def get_ori_data(self, data_path: str) -> np.ndarray:
        """
        Function:
            get raster data in ndarray from file
        Input:
            data_path: path of a single band raster file.
        Output:
            oriData: ndarray contains the required data.
        """
        try:
            dataset = gdal.Open(data_path)
            oriData = dataset.GetRasterBand(1).ReadAsArray()
        except Exception as e:
            self.my_logger.error(
                "open raster file error: {}, {}".format(e, data_path))
            return None
        return oriData

    def get_valid_data(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        *,
        nodata_value: float = np.nan,
        valid_range: float = 10000.,
    ) -> (np.ndarray, np.ndarray):
        """
        Function:
            old version - may have memory crushes
            get valid ground-truth data by applying mask
        Input:
            data:   ndarray of satellite image.
            mask:   ndarray of mask.
            *: optional parameters:
            nodata_value: values to be ignored in $ data.
            iss1:   boolean, if satellite is Sentinel-1, True; else, False.
            valid_range:
                    valid range of values in $data.
        Output:
            train_data: flattened array contains the required train data.
            mask_t:     flattened array contains the mask(label value).
        """
        mask_idx = np.where(mask > 0)
        train_data = data[mask_idx]
        mask_t = mask[mask_idx]

        if train_data.shape == mask_t.shape:
            print("train shape %d" % train_data.shape)
        else:
            raise Exception("get_valid_data(): shape not match! skip")

        #  get valid range and replace nodata value to nan
        train_data = train_data.astype(np.float)
        train_data[train_data == valid_range] = nodata_value

        return train_data.flatten(), mask_t.flatten()

    def get_valid_data_from_cloud(
            self,
            data: np.ndarray,
            mask: np.ndarray,
            qa_path: str,
            *,
            nodata_value: float=np.nan,
    ) -> (np.ndarray, np.ndarray):
        cloudmask = gdal.Open(qa_path)
        cloudmask_raster = cloudmask.GetRasterBand(1).ReadAsArray()
        print("cloud mask shape {}".format(cloudmask_raster.shape))
        # get cloud mask from qa file
        assert cloudmask_raster.shape == data.shape
        true_data = np.array([66, 130, 322, 386, 834, 898, 1346])
        print("True data value: {}".format(true_data))
        index = np.nonzero(np.isin(cloudmask_raster, true_data))
        print('index: ', index)
        if not list(index[0]):
            print("no valid value in tif")
            return None, None
        else:
            valid_data = np.full(cloudmask_raster.shape, nodata_value)
            valid_data[index] = 1
            # process original data by cloud mask
            valid_data = valid_data * data
            print("finish cloud masking")
        # process masked data by sample mask
        mask_idx = np.where(mask > 0)
        train_data = valid_data[mask_idx]
        mask_t = mask[mask_idx]

        if train_data.shape == mask_t.shape:
            print("train shape %d" % train_data.shape)
        else:
            raise Exception("get_valid_data(): shape not match! skip")

        train_data = train_data.flatten()
        mask_t = mask_t.flatten()
        train_data = train_data.astype(np.float)
        nan_idx1 = np.where(0. > train_data)[0]
        nan_idx2 = np.where(train_data > 10000.)[0]
        nan_idx = np.concatenate((nan_idx1, nan_idx2), axis=0)
        train_data[nan_idx] = np.nan
        # valid_idx = np.delete(np.arange(len(train_data)), nan_idx)
        # mask_t = mask_t[valid_idx]

        assert len(train_data) == len(mask_t)

        return train_data, mask_t

    def create_DEM_s1_mask(
            self,
            data,
            invalid_data: list,
    ):
        dem_mask = np.isin(data, invalid_data, invert=True)
        if not list(np.nonzero(dem_mask)[0]):
            raise Exception("no valid value in DEM")
            # TODO if DEM/S1 is all in valid, no need to do other mask
        return dem_mask

    def create_QA_mask(
            self,
            qa_path,
            data,
            invalid_range: list,
    ):
        """
        get mask from QA file, band data,
        :param qa_path:
        :param data:
        :param nodata_value:
        :return:
        """
        cloudmask = gdal.Open(qa_path)
        cloudmask_raster = cloudmask.GetRasterBand(1).ReadAsArray()
        print("cloud mask shape {}".format(cloudmask_raster.shape))
        # get cloud mask from qa file
        assert cloudmask_raster.shape == data.shape
        true_data = np.array([322, 386, 834, 898, 1346, 1])
        print("True data value: {}".format(true_data))
        qa_mask = np.isin(cloudmask_raster, true_data)
        qa_index = np.nonzero(qa_mask)
        print('index: ', qa_index)
        if not list(qa_index[0]):
            self.my_logger.debug("no valid value in QA")
            # TODO if QA is all in valid, no need to do other mask
            return False
        # get invalid data mask from band data tif, valid = (0, 10000]
        mask = np.logical_and(data > invalid_range[0], data <= invalid_range[1])
        w_mask = np.ma.array(data, mask=mask)
        qa_w_mask = w_mask.mask * qa_mask
        if not list(np.nonzero(qa_w_mask)[0]):
            self.my_logger.debug("no valid value in QA+tif")
            return False
        return qa_w_mask

    def binarize_label(self, label):
        # label keep list is given
        labels = self.label_keep_list
        new_label = np.full(label.shape, 0)
        for lb in labels:
            idx = np.where(label == lb)
            new_label[idx] = 1
        return new_label

    def stat_labels(self, label):
        labels_list = list(set(label))
        print("num of labels: {}, \nlabels: {}".format(
            len(labels_list), labels_list))

    def feature_norm_ref(self, data):
        """
        normalize features to reflectance (devide by 10000)
        """
        return data * 1.0 / self.ref_coef

    def go_get_mask_2npy(self) -> (np.ndarray, list, str):
        """
        Function:
            directly put raster data into array
        Warning:
            source list = [DEM, Landsat_8, Sentinel_1]
        Input:
        Output:
            ndarray contains all features and label, like this:
        final_feature = {
            Source1 = {
                'time' = ['20180101', '20180203', '20180521', ...],
                'Band1' = [
                    [p1-t1, p2-t1, p3-t1, ...],
                    [p1-t2, p2-t2, p3-t2, ...],
                    ...
                    [p1-tn, p1-tn, p1-tn, ...]
                ],
                ...
                'Bandn' = [
                    [p1-t1, p2-t1, p3-t1, ...],
                    [p1-t2, p2-t2, p3-t2, ...],
                    ...
                    [p1-tn, p1-tn, p1-tn, ...]
                ],
            }
            ...
            Sourcen = {}
        }
            npypath:      saved npz file path.

        NOTICE: train feature and train label are transposed before they are combined
                at #zzz_transpose!
        """

        # get geo information from the first item in dict
        first_raster_name = next(self.img_dict.walk())[-1]
        first_raster_name = first_raster_name[0]
        try:
            ds = gdal.Open(first_raster_name)
            geo_trans = ds.GetGeoTransform()
            prj_ref = ds.GetProjection()
            x_size = ds.RasterXSize
            y_size = ds.RasterYSize
            img_shape = [x_size, y_size]
        except Exception as e:
            self.my_logger.error(
                "open raster file error: {}".format(first_raster_name))
            return None, None, None

        mask_label_path = self.shp_label_path

        # get image projection
        raster_srs = osr.SpatialReference(wkt=prj_ref)
        assert(raster_srs.IsProjected()), "image projection info error!"
        epsg_raster = raster_srs.GetAttrValue("authority", 1)

        # get label shape projection
        shapef = ogr.Open(self.shp_label_path)
        lyr = shapef.GetLayer(0)
        shape_srs = lyr.GetSpatialRef()
        epsg_shape = shape_srs.GetAttrValue("authority", 1)

        # label shape reprojection
        if epsg_raster != epsg_shape:
            self.my_logger.info("reproject label shape")
            reprj_label_path = self.shp_label_path.replace(
                ".shp", "_reprj_{}.shp".format(epsg_raster))

            if os.path.exists(reprj_label_path):
                self.my_logger.info("reprojected shape file exist!")
            else:
                reprj_cmd_str = "ogr2ogr -t_srs EPSG:{} -s_srs EPSG:{} {} {}".format(
                    epsg_raster, epsg_shape, reprj_label_path, self.shp_label_path)
                self.my_logger.info("Shape reprojection command: {}".format(
                    reprj_cmd_str))
                process_status = subprocess.run(
                    [sub_str.replace(",", " ")
                     for sub_str in reprj_cmd_str.split()]
                )
                # check process status
                assert process_status.returncode == 0, "label shape reprojection failed!"

            mask_label_path = reprj_label_path

        # generate label mask
        self.my_logger.info("Generating label mask ...")
        mask, num_label, list_label = calc_mask_by_shape(
            mask_label_path,
            geo_trans,
            img_shape,
            specified_field=self.field_name,
            condition=None,
            mask_value=-1,
            flag_dlist=True,
            field_strict=True,
        )
        if mask is None:
            self.my_logger.error("calculating label mask error")
            return None, None, None
        print("There are {} polygons".format(num_label), np.max(mask))

        # get raster data
        self.my_logger.info("Getting raster data...")
        n_file = len(self.read_order_list)
        fn = 0

        qa_source = ['Landsat_8', 'Sentinel_2']
        final_feature = dict()
        for key in self.ori_dict.keys():
            final_feature[key] = dict()
        # loop read order list, read all raster datas
        train_lab = []
        dem_mask = []

        for ro in self.read_order_list:
            fn += 1
            print("{}/{} raster files:".format(fn, n_file))
            source = ro[0]  # "Sentinel_1", "Landsat_8", "DEM"
            ttime = ro[1]  # "20180111", "20180120", '20180120-1'
            print('data source => ', source)
            final_feature[source].setdefault('time', []).append(ttime[:8])
            b = 0
            QA_mask = []
            S1_mask = []
            RAS_DATA = []
            bad = 0
            for tif in self.img_dict[source][ttime]:
                bn = b + 1  # band_id, starts from 1
                print("mask {}/{} band:".format(bn, len(self.img_dict[source][ttime])))

                # read raster data band bn
                ras_data = self.get_ori_data(tif)
                if ras_data is None:
                    self.my_logger.error("get ori data error", tif)
                    return None, None, None

                RAS_DATA.append(ras_data)
                # get mask from DEM
                if 'D' in source:
                    if 'ASPECT' or 'SLOPE' in tif:
                        dem_mask.append(self.create_DEM_s1_mask(ras_data, [-9999.]))
                    else:
                        dem_mask.append(self.create_DEM_s1_mask(ras_data, [-32768.]))

                if source in qa_source:
                    pixel_qa_list = glob.glob(join(os.path.dirname(
                        self.img_dict[source][ttime][0]), '*_pixel_qa.tif'))
                    assert(len(pixel_qa_list) ==
                           1), "pixel qa file number error!"
                    qa_path = pixel_qa_list[0]
                    print('QA path {}'.format(qa_path))
                    qa_mask = self.create_QA_mask(qa_path, ras_data, [0., 10000.])
                    if np.nonzero(qa_mask)[0].shape[0] == 0:
                        bad = 1
                        break
                    QA_mask.append(qa_mask)

                else:
                    S1_mask.append(self.create_DEM_s1_mask(ras_data, [10000.]))
                b += 1
            print('finish mask => ', source)
            if bad == 1:
                self.my_logger.error(
                    "get valid data from {} in {}, stepping to next".format(source, ttime))
                final_feature[source]['time'].remove(ttime[:8])
                continue

            assert len(dem_mask) == 3
            DEM_mask = reduce(lambda x, y: x*y, dem_mask)

            mask = DEM_mask * mask
            mask_idx = np.where(mask > 0)
            train_lab = mask[mask_idx].flatten()
            bn = 0
            print(len(RAS_DATA))
            for data in RAS_DATA:
                bn += 1
                valid_data = np.full(data.shape, np.nan)
                if source in qa_source:
                    print('mask Landsat-8 ...')

                    assert len(QA_mask) == len(RAS_DATA)
                    final_QA_mask = reduce(lambda x, y: x * y, QA_mask)
                    try:
                        QA_DEM_mask = final_QA_mask * DEM_mask
                    except Exception:
                        self.my_logger.warning('{}-{} data shape unmatched!'.format(source, ttime))
                        break
                    index = np.nonzero(QA_DEM_mask)
                    print('index: ', index)
                elif 'Se' in source:
                    print('mask Sentinel-1 ...')
                    assert len(S1_mask) == len(RAS_DATA)
                    final_S1_mask = reduce(lambda x, y: x * y, S1_mask)
                    S1_DEM_mask = final_S1_mask * DEM_mask
                    index = np.nonzero(S1_DEM_mask)
                    print('index: ', index)
                else:
                    print('mask DEM ...')
                    index = np.nonzero(DEM_mask)

                valid_data[index] = 1
                valid_data = valid_data * data

                # process masked data by sample mask
                train_data = valid_data[mask_idx]

                if train_data.shape == train_lab.shape:
                    print("train shape %d" % train_data.shape)
                else:
                    raise Exception("get_valid_data(): shape not match! skip")

                train_data = train_data.flatten()

                band_key = 'Band_' + str(bn)
                self.my_logger.info(
                    "getting data from [{}->{}] [{}], shape:{}".format(
                        source, band_key, ttime, train_data.shape)
                )

                # append data to 2-D array: (points_num x times)
                final_feature[source].setdefault(
                    band_key, []).append(train_data)
                print(
                    "{source}->{band_key} feature shape: ->".format(
                        source=source, band_key=band_key),
                    np.array(final_feature[source][band_key]).shape
                )
        assert train_lab is not None
        # positive: 1; negative: 0
        if self.isbinarize:
            train_lab = self.binarize_label(np.array(train_lab))

        # get unique labels list
        self.stat_labels(train_lab)

        # save to npz
        npy_path = self.work_path + "TD_" + self.sample_label + "_extract.npz"
        if not os.path.exists(self.work_path):
            os.makedirs(self.work_path)
        np.savez(npy_path, features=final_feature, labels=train_lab)

        rolist_path = npy_path.replace(".npz", "_ro.json")
        save_json(rolist_path, self.read_order_list)
        self.my_logger.info("save train-data to NPZ success !")
        self.my_logger.info(npy_path)

        return final_feature, train_lab, npy_path
