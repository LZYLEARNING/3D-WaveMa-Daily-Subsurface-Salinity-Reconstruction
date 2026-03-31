import os
import scipy
import importlib
from utils import *
import pytorch_lightning as pl


class MyLightningModel(pl.LightningModule):
    def __init__(self, model_params, train_params, qyxx_params, new_params):
        super(MyLightningModel, self).__init__()

        # [new_params]
        self.H = new_params.get('H')
        self.W = new_params.get('W')
        self.result_path = new_params.get('result_path')
        self.combine_batches = new_params.get('combine_batches')
        self.SSS_name = new_params.get('SSS_name')
        self.tg_mask_path = new_params.get('tg_mask_path')
        self.types_mask_path = new_params.get('types_mask_path')
        self.if_pengzhang_edge = new_params.get('if_pengzhang_edge')
        self.pengzhang_size = new_params.get('pengzhang_size')
        self.range_type = new_params.get('range_type')
        self.output = []

        # [重新单拎出来的参数]
        # [qyxx_params]
        self.model_name = qyxx_params.get('model_name', "MambaIR_XB_DS")

        self.K = qyxx_params.get('K')
        self.region_ID = qyxx_params.get('region_ID', 1)

        self.h_win = qyxx_params.get('h_win', 28)
        self.w_win = qyxx_params.get('w_win', 24)
        self.h_stride = qyxx_params.get('h_valtest_stride', 10)
        self.w_stride = qyxx_params.get('w_valtest_stride', 12)
        # self.h_win = 16
        # self.w_win = 20
        # self.h_stride = 4
        # self.w_stride = 5

        # [train_params]
        self.if_double = train_params.get('if_double', True)
        self.layers = train_params.get('layers', 41)
        self.tg_mask = loadmat(self.tg_mask_path)['mask'][:-1, :-1, 0:self.layers].transpose(2, 0, 1)
        self.tg_mask = torch.tensor(self.tg_mask, dtype=torch.int32)
        if self.if_pengzhang_edge:
            self.types_mask, self.new_mask = types_mask_pool_pengzhang(dir=self.types_mask_path,
                                                             h_win=self.h_win,
                                                             w_win=self.w_win,
                                                             s=(self.h_stride, self.w_stride),
                                                             region_ID=self.region_ID,
                                                             pengzhang_size=self.pengzhang_size)
        elif self.range_type == 'Global':
            self.types_mask, self.new_mask = types_mask_pool(dir=self.types_mask_path,
                                                             h_win=self.h_win,
                                                             w_win=self.w_win,
                                                             s=(self.h_stride, self.w_stride),
                                                             region_ID=self.region_ID)
        else:
            self.types_mask, self.new_mask = types_mask_pool_partition_predict(dir=self.types_mask_path,
                                                             h_win=self.h_win,
                                                             w_win=self.w_win,
                                                             s=(self.h_stride, self.w_stride),
                                                             region_ID=self.region_ID)

        self.types_num = torch.max(self.types_mask)
        self.types_idx = [[]] * int(self.types_num)
        for t in range(self.types_mask.shape[0] * self.types_mask.shape[1]):
            type_idx = int(self.types_mask[
                               int(t / self.types_mask.shape[1]), int(t % self.types_mask.shape[1])])
            if type_idx != 0:
                self.types_idx[type_idx - 1].append(t)
        class_name = "BuildModel"
        module = importlib.import_module(f"model_{self.model_name}")
        Model = getattr(module, class_name)
        print(f"成功导入 model_{self.model_name} 的 {class_name} 类")

        self.model = torch.nn.ModuleList([Model(model_params) for _ in range(int(self.types_num))])
        print(self.model)

        # for module_name, module_structure in self.model.model[0].named_children():
        #     print(f"module name: {module_name}")
        #     print(f"module structure: {module_structure}")
        #     for layer_name, layer_param in module_structure.named_parameters():
        #         # 输出该参数所属的层和该参数的名称
        #         print(f"layer name: {layer_name}, "
        #               f"belongs to layer: {module_name}, "
        #               f"Parameter shape: {layer_param.size()}, "
        #               f"requires_grad: {layer_param.requires_grad}")

    def forward(self, inputs, B, cbs):
        types_ops = [[] for _ in range(int(self.types_num))]
        for tp in range(int(self.types_num)):  # 估计10类, self.types_num = 10 == 10个并行的模型
            ip = torch.concat(inputs[tp], dim=0)
            allcblist = list(range(math.ceil(ip.shape[0] / (B * cbs))))
            for cb_idx in allcblist:  # 估计1到2次
                cb_start = cb_idx * cbs * B
                cb_end = min((cb_idx + 1) * cbs * B, ip.shape[0])
                types_ops[tp].append(self.model[tp](ip[cb_start: cb_end, :, :, :]))
        return types_ops

    def predict_step(self, batch):
        self.eval()
        current_date = batch[1][0]
        batch = batch[0]
        print("current_date:", current_date)

        # [气候态]
        tg_clim = batch[0]
        tg_clim[tg_clim != tg_clim] = 1
        if not self.if_double:
            tg_clim = tg_clim.float()
        op_shape = [tg_clim.shape[0], self.layers, self.H, self.W]  # 创建一个与 batch[0] 尺寸一致的零张量
        B = op_shape[0]

        # [统计量]
        SS_fz = batch[1][:, 0:1, :, :, :].squeeze(1).contiguous()
        SS_fz[SS_fz != SS_fz] = 1
        SS_fm = batch[1][:, 1:2, :, :, :].squeeze(1).contiguous()
        SS_fm[SS_fm != SS_fm] = 1
        if self.if_double:
            tg_stc = [SS_fz, SS_fm]
        else:
            tg_stc = [SS_fz.float(), SS_fm.float()]

        # [输入]
        types_ips = batch[2]
        tg_mask_predict = (self.tg_mask.unsqueeze(0).expand(B, -1, -1, -1)).to(tg_clim.device)

        # [输出-Forward]
        op_device = tg_clim.device
        op_dtype = tg_clim.dtype
        types_ops = self(types_ips, B, self.combine_batches)

        # [输出-复原]
        output = torch.zeros(op_shape, device=op_device, dtype=op_dtype)
        output_norm = torch.zeros(op_shape, device=op_device, dtype=op_dtype)
        gaussian_map = get_gaussian((self.h_win, self.w_win), B, self.layers, 1.0 / 8, op_dtype, op_device)
        for tp in range(int(self.types_num)):
            op = torch.concat(types_ops[tp], dim=0)
            tidx = 0
            for ti in self.types_idx[tp]:
                top = int(self.h_stride * int(ti / self.types_mask.shape[1]))
                left = int(self.w_stride * int(ti % self.types_mask.shape[1]))
                patch = op[B * tidx: B * (tidx + 1), :, :, :]
                patch *= gaussian_map
                output[:, :, top:top + self.h_win, left:left + self.w_win] += patch
                output_norm[:, :, top:top + self.h_win, left:left + self.w_win] += gaussian_map
                tidx = tidx + 1

        # [保存结果]
        # [不分区]
        if self.range_type == 'Global':
            output /= output_norm
            # [数据后处理-反归一化 & 加气候态 & 网格掩膜规整]
            output = ((output * tg_stc[1]) + tg_stc[0]) + tg_clim
            output[tg_mask_predict == 0] = torch.nan

            # [将预测结果添加到类变量中]
            os.makedirs(self.result_path, exist_ok=True)
            file_path = f'{self.result_path}/{current_date}'
            output = output.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            scipy.io.savemat(file_path, {'origin': output})

        # [分区][在matlab中进行后处理]
        else:
            # [将预测结果添加到类变量中]
            os.makedirs(self.result_path, exist_ok=True)
            os.makedirs(f'{self.result_path}_gaussian', exist_ok=True)
            file_path = f'{self.result_path}/{current_date}'
            file_path_gaussian = f'{self.result_path}_gaussian/{current_date}'
            output = output.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            scipy.io.savemat(file_path, {'origin': output})
            output_norm = output_norm.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            scipy.io.savemat(file_path_gaussian, {'origin': output_norm})

