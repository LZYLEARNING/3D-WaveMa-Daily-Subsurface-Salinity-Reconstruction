import os
from utils import *
from scipy.io import loadmat
from torch.utils.data import Dataset


class MyDataset_predict(Dataset):
    def __init__(self,
                 if_double,
                 predict_start_date,
                 predict_end_date,
                 statistics_time,
                 range_type,
                 region_ID,
                 combine_batches,
                 types_mask_path,
                 if_pengzhang_edge,
                 pengzhang_size,
                 h_win,
                 w_win,
                 h_stride,
                 w_stride,
                 gyh_region,
                 gyh_type,
                 clim_mode,
                 data_path,
                 region,
                 variables,
                 decorr_days,
                 layers,
                 max_insitu_num):

        # [是否双精度]
        self.if_double = if_double
        # [统计时段]
        self.statistics_time = statistics_time
        # [是否分区]
        self.range_type = range_type

        # [扩充倍数]
        self.combine_batches = combine_batches
        # [patch的窗口大小与步长]
        self.h_win = h_win
        self.w_win = w_win
        self.h_stride = h_stride
        self.w_stride = w_stride
        # [分类掩码]
        if if_pengzhang_edge:
            self.types_mask, self.new_mask = types_mask_pool_pengzhang(dir=types_mask_path,
                                                             h_win=self.h_win,
                                                             w_win=self.w_win,
                                                             s=(self.h_stride, self.w_stride),
                                                             region_ID=region_ID,
                                                             pengzhang_size=pengzhang_size)
        elif self.range_type == 'Global':
            self.types_mask, self.new_mask = types_mask_pool(dir=types_mask_path,
                                                             h_win=self.h_win,
                                                             w_win=self.w_win,
                                                             s=(self.h_stride, self.w_stride),
                                                             region_ID=region_ID)
        else:
            self.types_mask, self.new_mask = types_mask_pool_partition_predict(dir=types_mask_path,
                                                             h_win=self.h_win,
                                                             w_win=self.w_win,
                                                             s=(self.h_stride, self.w_stride),
                                                             region_ID=region_ID)


        # [分类数]
        self.types_num = torch.max(self.types_mask)
        # [裁剪的小patch总数]
        mask_crop_num = self.types_mask.shape[0] * self.types_mask.shape[1]
        # [所有类型的大区域中的小patch的位置索引]
        self.types_idx = [[]] * int(self.types_num)
        for t in range(mask_crop_num):
            type_idx = int(self.types_mask[int(t / self.types_mask.shape[1]), int(t % self.types_mask.shape[1])])
            if type_idx != 0:
                self.types_idx[type_idx - 1].append(t)
        # [重构层数]
        self.layers = layers  # 去相关尺度的天数包含当天
        # [去相关天数]
        self.decorr_days = decorr_days  # 去相关尺度的天数包含当天
        # [变量个数]
        self.var_num = len(variables) - 2
        # [文件列表]
        self.files_list = _get_file_list(data_path + '/' + region + '/origin/all_files.mat')  # 某set的文件名
        predict_start_index = self.files_list.index(predict_start_date+'.mat')  # 获取predict_start的索引
        predict_end_index = self.files_list.index(predict_end_date+'.mat')+1  # 因为要保证最后的一个mat文件要取到
        self.files_list = self.files_list[predict_start_index - self.decorr_days + 1:predict_end_index]
        # [变量]
        self.variables_dir = [data_path + '/' + region + '/origin' + '/' + var for var in variables]
        # [统计量]
        self.statistics_dir = [data_path + '/' + region + '/statistic' + self.statistics_time + '/' + var for var in variables]
        # [归一化范围与类型]
        self.gyh_region = gyh_region
        self.gyh_type = gyh_type
        # [气候态]
        self.clim_mode = clim_mode
        self.clims_dir = [data_path + '/' + region + '/clim' + '/' + var for var in variables]
        # [Insitu]
        self.max_insitu_num = max_insitu_num
        # [检查]
        print("Dataset加载的起始索引:", predict_start_index)
        print("Dataset加载的结束索引:", predict_end_index)
        print("Dataset加载的起始日期:", predict_start_date)
        print("Dataset加载的结束日期:", predict_end_date)
        print("实际推理天数:", predict_end_index - predict_start_index, "天")
        print("Dataset加载的variables_dir", self.variables_dir)
        print("Dataset加载的statistics_dir", self.statistics_dir)
        print("Dataset加载的clims_dir", self.clims_dir)
        print("Dataset加载的clim_mode", self.clim_mode)
        print("Dataset加载的gyh_region", self.gyh_region)
        print("Dataset加载的gyh_type", self.gyh_type)

    def __len__(self):
        # __len__ 方法返回的数据集的大小是 len(self.files_list) - self.decorr_days + 1。
        # 这意味着每个 epoch 中的所有 index 范围是从 0 到 len(self.files_list) - self.decorr_days
        return len(self.files_list) - self.decorr_days + 1

    def __getitem__(self, index):

        # [索引校对]
        global fz, fm
        index = index + self.decorr_days - 1

        # [批次]
        batch = [None] * 2  # batch格式 = [tg1_气候态, tg1, tg2, tg1_统计量, ipx9, ip_统计量x9] = 23个
        ip_batch = [None] * self.var_num * 2

        # [clim]
        # (41, 280, 600) ==> (B, 41, 280, 600)
        if self.clim_mode == "Minus":
            clim = loadmat(self.clims_dir[0] + '/'
                           + self.files_list[index][4:6]
                           + '.mat')['origin'][:-1, :-1, 0:self.layers].transpose(2, 0, 1)
        else:
            clim = np.zeros((1, 1, 1))
        batch[0] = clim
        del clim

        # [变量]
        ip_idx = 0
        for variable_dir, clim_dir in zip(self.variables_dir, self.clims_dir):
            if os.path.basename(variable_dir) != 'SS_GREP_mnstd' and os.path.basename(variable_dir) != 'SS_EN4_profiles_unfixed':
                ip_variables = []
                for d in range(self.decorr_days):
                    file_name = self.files_list[index - self.decorr_days + 1 + d]
                    ip_variable = loadmat(variable_dir + '/' + file_name)['origin']
                    if self.clim_mode == "Minus":  # 减去气候态
                        clim = loadmat(clim_dir + '/' + file_name[4:6] + '.mat')['origin']
                        ip_variable = np.subtract(ip_variable, clim)
                    ip_variable = np.expand_dims(ip_variable[:-1, :-1], axis=0)
                    ip_variables.append(ip_variable)
                ip_variables = np.concatenate(ip_variables, axis=0)
                ip_batch[ip_idx] = ip_variables
                del ip_variables
                ip_idx = ip_idx + 1

        # [统计量_File / Point]
        # [统计量_File]
        ip_tj_idx = self.var_num
        if self.gyh_region == 'File':
            for statistic_dir in self.statistics_dir:
                if os.path.basename(statistic_dir) == 'SS_EN4_profiles_unfixed':
                    continue
                statistic = loadmat(statistic_dir + "/" + self.clim_mode + ".mat")[self.clim_mode]
                if os.path.basename(statistic_dir) == 'SS_GREP_mnstd':
                    if self.gyh_type == 'Norm':
                        fz = np.expand_dims(statistic['mean'][0][0][:-1, :-1, 0:self.layers].transpose(2, 0, 1), axis=0)
                        fm = np.expand_dims(statistic['std_zt'][0][0][:-1, :-1, 0:self.layers].transpose(2, 0, 1), axis=0)
                    elif self.gyh_type == 'MinMax':
                        fz = np.expand_dims(statistic['min'][0][0][:-1, :-1, 0:self.layers].transpose(2, 0, 1), axis=0)
                        fm = np.expand_dims(statistic['max'][0][0][:-1, :-1, 0:self.layers].transpose(2, 0, 1), axis=0) - fz
                    tg_tsc = np.concatenate([fz, fm], axis=0)
                    batch[1] = tg_tsc
                    del tg_tsc
                else:
                    if self.gyh_type == 'Norm':
                        fz = np.expand_dims(statistic['mean'][0][0][:-1, :-1], axis=(0, 1))
                        fm = np.expand_dims(statistic['std_zt'][0][0][:-1, :-1], axis=(0, 1))
                    elif self.gyh_type == 'MinMax':
                        fz = np.expand_dims(statistic['min'][0][0][:-1, :-1], axis=(0, 1))
                        fm = np.expand_dims(statistic['max'][0][0][:-1, :-1], axis=(0, 1)) - fz
                    ip_tsc = np.concatenate([fz, fm], axis=0)
                    ip_batch[ip_tj_idx] = ip_tsc
                    del ip_tsc
                    ip_tj_idx = ip_tj_idx + 1

        # [统计量_Point]
        # (2, 50, 1, 1)  & (2, 1, 1, 1)
        elif self.gyh_region == 'Point':
            for statistic_dir in self.statistics_dir:
                if os.path.basename(statistic_dir) == 'SS_EN4_profiles_unfixed':
                    continue
                statistic = loadmat(statistic_dir + "/" + self.clim_mode + ".mat")[self.clim_mode]
                if os.path.basename(statistic_dir) == 'SS_GREP_mnstd':
                    if self.gyh_type == 'Norm':
                        fz = np.expand_dims(statistic['all_mean'][0][0][0:self.layers, :].transpose(1, 0), axis=(-2, -1))
                        fm = np.expand_dims(statistic['all_std_zt'][0][0][0:self.layers, :].transpose(1, 0), axis=(-2, -1))
                    elif self.gyh_type == 'MinMax':
                        fz = np.expand_dims(statistic['all_min'][0][0][0:self.layers, :].transpose(1, 0), axis=(-2, -1))
                        fm = np.expand_dims(statistic['all_max'][0][0][0:self.layers, :].transpose(1, 0), axis=(-2, -1)) - fz
                    tg_tsc = np.concatenate([fz, fm], axis=0)
                    batch[1] = tg_tsc
                    del tg_tsc
                else:
                    if self.gyh_type == 'Norm':
                        fz = np.expand_dims(statistic['all_mean'][0][0], axis=(-2, -1))
                        fm = np.expand_dims(statistic['all_std_zt'][0][0], axis=(-2, -1))
                    elif self.gyh_type == 'MinMax':
                        fz = np.expand_dims(statistic['all_min'][0][0], axis=(-2, -1))
                        fm = np.expand_dims(statistic['all_max'][0][0], axis=(-2, -1)) - fz
                    ip_tsc = np.concatenate([fz, fm], axis=0)
                    ip_batch[ip_tj_idx] = ip_tsc
                    del ip_tsc
                    ip_tj_idx = ip_tj_idx + 1

        # [全部patch一起处理] [仅处理ips]
        types_ips = [[]] * int(self.types_num)
        types_idx_tmp = [[]] * int(self.types_num)
        for ty in range(int(self.types_num)):
            # [全部patch一起处理]
            types_idx_tmp[ty] = self.types_idx[ty]
            for t in types_idx_tmp[ty]:
                p = [0 for _ in range(4)]
                ips = data_transform_ips(if_double=self.if_double,
                                         gyh_region=self.gyh_region,
                                         data=ip_batch,
                                         h_win=self.h_win, w_win=self.w_win,
                                         h_stride=self.h_stride, w_stride=self.w_stride,
                                         w_size=self.types_mask.shape[1],
                                         p=p, t=t)
                types_ips[ty].append(ips)
        batch.append(types_ips)
        del ip_batch, types_ips, types_idx_tmp

        return batch, self.files_list[index]


def _get_file_list(path):
    files_struct = loadmat(path)['files']['name']
    files_list = []
    for i in range(len(files_struct)):
        files_list.append(files_struct[i][0][0])
    return files_list
