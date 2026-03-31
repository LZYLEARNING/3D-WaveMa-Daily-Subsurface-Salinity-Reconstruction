import os
import torch
import pytorch_lightning as pl
from predict_data_lightning import MyDataModule
from predict_model_lightning import MyLightningModel


if __name__ == '__main__':
    # [可动][修改参数] ================================================================================================ #
    range_type = 'Global_TL_Local_3sub'
    gyh_type = 'file-minus'
    ckpt_name = r'PRL_30_2025y_08m16d_09h19m_TL0_for_Subregion_from___TIME_2025y_03m09d_21h38m_MODEL_My_REGION1-3_CKPT_0.099562.ckpt'
    model_name = 'My_1-3'
    if_pengzhang_edge = False  # 推理局部是将局部的边缘向外膨胀一圈
    pengzhang_size = 7  # 推理局部是将局部的边缘向外膨胀一圈的范围尺度
    if_all = False  # 直接拿局部训练的模型推理全局结果
    SSS_name = 'S5R2'  # GREP_mnstd, CCI, SMOS, SMAP
    predict_start_date = '20220101'
    predict_end_date = '20221231'
    combine_batches = 100
    H = 21
    W = 30
    # ================================================================================================================ #


    # [自动][生成seed和path] =========================================================================================== #
    seed = 32  # 根本参数, 不许动
    data_path = 'D:/Project/SSR/data'  # PC端的数据来源，与K和region_ID无关，保持不变即可
    result_path = f'D:/Project/SSR/data/PO/result'
    if range_type == 'Global':
        result_path = f'{result_path}/{gyh_type}/{model_name}'
    else:
        if if_pengzhang_edge:
            result_path = f'{result_path}/{gyh_type}/#{range_type}_pengzhang_edge_{pengzhang_size}/{model_name}'
        elif if_all:
            result_path = f'{result_path}/{gyh_type}/#{range_type}_all/{model_name}'
        else:
            result_path = f'{result_path}/{gyh_type}/#{range_type}/{model_name}'
    ckpt_path = f'D:/Project/SSR/data/PO/result/#CKPT/{gyh_type}/#{range_type}/{ckpt_name}'
    ckpt_files = [os.path.join(ckpt_path, file) for file in os.listdir(ckpt_path) if file.endswith('.ckpt')]

    # [自动][创建保存结果的文件夹]
    if not os.path.exists(result_path):
        os.makedirs(result_path)  # 如果不存在，创建文件夹
    else:
        # 如果文件夹已存在，添加后缀（1），直到找到一个可用的文件夹名
        base_path = result_path
        counter = 1
        while os.path.exists(result_path):
            result_path = f"{base_path}_{counter}"
            counter += 1
        os.makedirs(result_path)  # 创建新的唯一文件夹

    # [自动][声明改输出来自的模型与版本]
    file_path = os.path.join(result_path, f'#CKPT#_#{range_type}_{ckpt_name}.txt')
    with open(file_path, 'w') as file:
        pass
    # ================================================================================================================ #


    # [推理][分区循环] ================================================================================================= #
    for ckpt_file in ckpt_files:
        # [获取参数]
        # [更新new_params]
        og_qyxx_params = torch.load(ckpt_file)['hyper_parameters']['qyxx_params']
        og_data_params = torch.load(ckpt_file)['hyper_parameters']['train_params']['data_params']
        region = og_data_params['region']
        K = og_qyxx_params['K']
        region_ID = og_qyxx_params['region_ID']
        tg_mask_path = f"{data_path}/{region}/mask/mask_3d.mat"
        if if_all:
            types_mask_path = f"{data_path}/{region}/mask/mask_2d_{1}.mat"
        else:
            types_mask_path = f"{data_path}/{region}/mask/mask_2d_{K}.mat"
        new_params = {
            "result_path": result_path,
            "combine_batches": combine_batches,
            "SSS_name": SSS_name,
            "tg_mask_path": tg_mask_path,
            "types_mask_path": types_mask_path,
            "if_pengzhang_edge": if_pengzhang_edge,
            "pengzhang_size": pengzhang_size,
            "range_type": range_type,
            "H": H,
            "W": W}

        # [更新data_params]
        variables = og_data_params['variables']
        variables[2] = f"SSS_{SSS_name}"
        pd_data_params = {
            "if_double": og_data_params['if_double'],
            "predict_start_date": predict_start_date,
            "predict_end_date": predict_end_date,
            "statistics_time": og_data_params['statistics_time'],
            "range_type": range_type,
            "region_ID": region_ID,
            "combine_batches": combine_batches,
            "types_mask_path": types_mask_path,
            "if_pengzhang_edge": if_pengzhang_edge,
            "pengzhang_size": pengzhang_size,
            # "h_win": 28,  # og_data_params['h_win'],
            # "w_win": 20,  # og_data_params['w_win'],
            # "h_stride": 4,  # og_data_params['h_valtest_stride'],
            # "w_stride": 5,  # og_data_params['w_valtest_stride'],
            "h_win": og_data_params['h_win'],
            "w_win": og_data_params['w_win'],
            "h_stride": og_data_params['h_valtest_stride'],
            "w_stride": og_data_params['w_valtest_stride'],
            "gyh_region": og_data_params['gyh_region'],
            "gyh_type": og_data_params['gyh_type'],
            "clim_mode": og_data_params['clim_mode'],
            "data_path": data_path,
            "region": region,
            "variables": variables,
            "decorr_days": og_data_params['decorr_days'],
            "layers": og_data_params['layers'],
            "max_insitu_num": og_data_params['max_insitu_num'],
            "batch_size": 1,
            "num_workers": 4}
        data = MyDataModule(pd_data_params)
        trainer = pl.Trainer(deterministic='warn', accelerator="gpu", devices=1)

        # [打印确认]
        print("当前处理的ckpt:", ckpt_file)
        print("划分子区域个数为:", K)
        print("处理的子区域为:", og_qyxx_params['region_ID'])

        # [实例化]
        pl.seed_everything(seed, workers=True)
        # [赋予参数][将参数加载到模型中，赋予其weight, bias, hparams]
        model = MyLightningModel.load_from_checkpoint(ckpt_file, new_params=new_params)  # 加载并覆盖指定参数

        # [推理结果]
        trainer.predict(model, datamodule=data, return_predictions=True)
        # ============================================================================================================ #
