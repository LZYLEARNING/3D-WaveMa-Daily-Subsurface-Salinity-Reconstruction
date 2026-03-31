import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data_set_predict import MyDataset_predict


class MyDataModule(pl.LightningDataModule):
    def __init__(self, kwargs):
        super().__init__()
        self.dataset = None
        self.if_double = kwargs.get('if_double', 28)
        self.predict_start_date = kwargs.get('predict_start_date', 28)
        self.predict_end_date = kwargs.get('predict_end_date', 28)
        self.statistics_time = kwargs.get('statistics_time', '_end2021')
        self.range_type = kwargs.get('range_type', 28)
        self.region_ID = kwargs.get('region_ID', 28)
        self.h_win = kwargs.get('h_win', 28)
        self.w_win = kwargs.get('w_win', 24)
        self.h_stride = kwargs.get('h_stride', 5)
        self.w_stride = kwargs.get('w_stride', 6)
        self.types_mask_path = kwargs.get('types_mask_path')
        self.if_pengzhang_edge = kwargs.get('if_pengzhang_edge')
        self.pengzhang_size = kwargs.get('pengzhang_size')
        self.combine_batches = kwargs.get('combine_batches')
        self.gyh_region = kwargs.get('gyh_region')  # or "Point"
        self.gyh_type = kwargs.get('gyh_type')  # or "MinMax"
        self.clim_mode = kwargs.get('clim_mode')  # or "minus"
        self.data_path = kwargs.get('data_path')
        self.region = kwargs.get('region')
        self.variables = kwargs.get('variables')
        self.decorr_days = kwargs.get('decorr_days')
        self.layers = kwargs.get('layers')
        self.max_insitu_num = kwargs.get('max_insitu_num')
        self.batch_size = kwargs.get('batch_size')
        self.num_workers = kwargs.get('num_workers')
        print("Data_Lightning加载的variables", self.variables)
        print("Data_Lightning加载的decorr_days延迟天数", self.decorr_days)

    def setup(self, stage=None):
        self.dataset = MyDataset_predict(if_double=self.if_double,
                                         predict_start_date=self.predict_start_date,
                                         predict_end_date=self.predict_end_date,
                                         statistics_time=self.statistics_time,
                                         range_type=self.range_type, region_ID=self.region_ID,
                                         combine_batches=self.combine_batches, types_mask_path=self.types_mask_path,
                                         if_pengzhang_edge=self.if_pengzhang_edge,
                                         pengzhang_size=self.pengzhang_size,
                                         h_win=self.h_win, w_win=self.w_win, h_stride=self.h_stride,
                                         w_stride=self.w_stride, gyh_region=self.gyh_region, gyh_type=self.gyh_type,
                                         clim_mode=self.clim_mode, data_path=self.data_path, region=self.region,
                                         variables=self.variables, decorr_days=self.decorr_days, layers=self.layers,
                                         max_insitu_num=self.max_insitu_num)

    def predict_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          pin_memory=False)
