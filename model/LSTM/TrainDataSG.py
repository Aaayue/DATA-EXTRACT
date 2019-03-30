import logging
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import datetime
import os


class TrainDataSG:
    my_logger = logging.getLogger(__qualname__)

    def __init__(self,
                 file,
                 save_path,
                 year='2018',
                 start_day='0101',
                 end_day='1231',
                 sg_window=17,
                 sg_polyorder=1,
                 quantity=2000
                 ):
        self.year = year
        self.start_day = year + start_day
        self.end_day = year + end_day
        self.sg_window = sg_window
        self.sg_polyorder = sg_polyorder
        self.file = file
        self.save_path = save_path
        self.quantity = quantity
        self.feature, self.label = self._load_npz()
        # create new time and to order
        nidx = pd.date_range(self.start_day, self.end_day,
                             freq="5D")  # time index
        new_time_tmp = nidx.to_pydatetime()
        new_time = [tmp.strftime("%Y%m%d") for tmp in new_time_tmp]  # time str
        self.new_time_order = list(map(self._ymd_to_jd, new_time))

    def _load_npz(self):
        try:
            data = np.load(self.file)
            feature = data['features'].tolist()
            label = data['labels'].tolist()
            return feature, label
        except Exception as e:
            print('Load {} error: {}'.format(self.file, e))
            return None

    def _ymd_to_jd(self, str_time):
        fmt = "%Y%m%d"
        dt = datetime.datetime.strptime(str_time, fmt)
        tt = dt.timetuple()
        return tt.tm_yday

    def _interpolate_and_sg(self, original_data, original_time):
        """
        data_series = [v1, v2, v3, ...]
        original_time = [t1, t2, t3, ...]
        SG smoothing
        """
        original_time_order = list(
            map(self._ymd_to_jd, original_time))  # YMD to order

        # interp
        valid_idx = np.where(~np.isnan(original_data))
        original_data = original_data[valid_idx]
        original_time_order = np.array(original_time_order)[valid_idx]

        inter_data = np.interp(self.new_time_order,
                               original_time_order, original_data)

        # using SG to filter the data, window_length = 17, polyorder = 1
        result_sg = savgol_filter(
            inter_data,
            window_length=self.sg_window,
            polyorder=self.sg_polyorder,
            mode='nearest'
        )

        return list(result_sg)

    def get_valid_time_sequence(
            self,
            data_dict, source, it,
            process_dict=None,
            bad_list=[],
    ):
        """
        function to list the ivalid time-series data index
        data_dict: full extracted data dictionary
        source: data source
        process_dict: not neccessary for SG masking, but needed for SG filtering
        bad_list: only needed for SG filtering
        """
        source_data = data_dict[source]
        band_list = [b for b in source_data.keys()]
        time_list = source_data['time']
        band_list.remove('time')
        print(band_list)
        for band in band_list:
            band_data = np.array(source_data[band]).T
            # print(band_data.shape)
            if not isinstance(process_dict, dict):
                print('MASK band: {} in source: {}'.format(band, source))
                tmp_data = list(band_data)[
                    (it * self.quantity): (it + 1) * self.quantity]
                print(len(tmp_data))
                for n, data_series in enumerate(tmp_data):
                    if source != 'DEM':
                        if np.isnan(data_series).sum() == len(data_series):
                            bad_list.append(n)
                            continue
                    else:
                        continue

            else:   # flag == 'sg'
                print('PROCESS band: {} in source: {}'.format(band, source))
                new_key = source + '-' + band
                tmp_data = list(band_data)[
                    (it * self.quantity): (it + 1) * self.quantity]
                print(len(tmp_data))
                for n, data_series in enumerate(tmp_data):
                    if n in bad_list:
                        continue
                    else:
                        if source == 'DEM':
                            res = np.full(
                                (1, len(self.new_time_order)), data_series)
                            res = res.tolist()[0]
                        else:
                            res = self._interpolate_and_sg(
                                data_series, time_list)
                        process_dict.setdefault(
                            new_key, []).append(res)
                print('finish band: {} in source: {} \n'.format(band, source))
        return process_dict, bad_list

    def single_run(self, itera, data_d):
        """

        :param iter: number of chunk
        :param data_d: full feature dictionary
        :return:w
        """
        precess_dict = dict()
        # bad_end = 0
        bad_list = []
        # TODO: get source list out of code
        source_list = ['Optical', 'Sentinel_1', 'DEM']
        for source in source_list:
            if source not in list(data_d.keys()):
                continue
            print("* * * * start SG mask * * * *")
            print('process source {} \n'.format(source))
            _, single_source_bad_list = self.get_valid_time_sequence(
                data_d, source, itera, process_dict=None, bad_list=bad_list)
            bad_list += single_source_bad_list

        bad_list = list(set(bad_list))
        if len(bad_list) == self.quantity:
            self.my_logger.error("Invalid value in this = = CHUNK = =")
            return None, bad_list
        else:
            print('Invalid data index in chunk {}: ({}) '.format(
                itera + 1, len(bad_list)))
            print(bad_list)
            for source in source_list:
                if source not in list(data_d.keys()):
                    continue
                print("= = = = start SG = = = =")
                print('process source {} \n'.format(source))
                precess_dict, _ = self.get_valid_time_sequence(
                    data_d, source, itera, process_dict=precess_dict, bad_list=bad_list)

                print('finish source {}'.format(source))
            return precess_dict, bad_list

    def batch_run(self):
        """

        :param quantity: chunk size
        :return: 2-D array
        [
            [p1-t1, p1-t2, ...],
            [p2-t1, p2-t2, ...],
            ...
        ]
        """

        length = len(self.label)
        iter_total = int(length / self.quantity) + 1
        for n in range(iter_total):
            iters = n + 1
            print('PROCESS {}/{} slide of data \n'.format(iters, iter_total))
            print('CHUNK SIZE {}'.format(self.quantity))
            tmp_lab = list(self.label)[
                (n * self.quantity): (n + 1) * self.quantity]
            precess_dict, invalid_list = self.single_run(n, self.feature)
            # invalid_list = list(set(invalid_list))
            print(len(invalid_list), len(tmp_lab))
            if len(invalid_list) >= len(tmp_lab):
                continue

            else:
                valid_list = [k for k in range(
                    len(tmp_lab)) if k not in invalid_list]
                tmp_lab = list(np.array(tmp_lab)[valid_list])
                print('length of label ', len(tmp_lab))
                save_file = os.path.join(
                        self.save_path,
                        os.path.basename(self.file).split('.')[0]
                        + "_"
                        + self.start_day[-4:]
                        + "_"
                        + self.end_day[-4:]
                        + "_"
                        + str(self.sg_window)
                        + "_"
                        + str(self.sg_polyorder)
                        + "_"
                        + str(self.quantity)  # noqa :F405
                        + "_"
                        + str(n + 1)
                        + ".npz"
                )
                np.savez(save_file, features=precess_dict, labels=tmp_lab)
            print('{}/{} DONE! \n'.format(iters, iter_total))
        print('HERO: ALL DONE')
