import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Callable
from timm.models.layers import DropPath, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import repeat


class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=2):  # 16-2
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class Conv_CA(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=2):  # 16-2
        super(Conv_CA, self).__init__()
        self.conv_ca = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor))

    def forward(self, x):
        return self.conv_ca(x)


class VSSM_H(nn.Module):  # 对应论文(b)Vision State Space Module (VSSM)
    def __init__(
            self,
            if_double,
            d_model,
            d_state=16,
            d_conv=3,  # 使用默认
            expand=2.,  # 就是模型导入的默认参数 mlp_ratio
            dt_rank="auto",  # 使用默认
            dt_min=0.001,  # 使用默认
            dt_max=0.1,  # 使用默认
            dt_init="random",  # 使用默认
            dt_scale=1.0,  # 使用默认
            dt_init_floor=1e-4,  # 使用默认
            dropout=0.,  # 使用默认
            conv_bias=True,  # 使用默认
            bias=False,  # 使用默认
            device=None,  # 使用默认
            dtype=None, ):  # 使用默认
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.if_double = if_double

        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 2) if dt_rank == "auto" else dt_rank  # 16-2

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.x_proj = (nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                       nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                       nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                       nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),)
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),)
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.if_double, self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    # 全都是默认参数
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(if_double, d_state, d_inner, copies=1, device=None, merge=True):
        # merge=True就是说4个扫描方向结果合起来，也不新增维度，就是同纬度拼起来
        if if_double:
            dtype = torch.double
        else:
            dtype = torch.float

        A = repeat(
            torch.arange(1, d_state + 1, dtype=dtype, device=device),
            "n -> d n",  # n = d_state, d = d_inner
            d=d_inner,  # [d, n] = [30, 16]
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)  # [r, d, n] = [4, 30, 16]
            if merge:
                A_log = A_log.flatten(0, 1)  # [r*d, n] = [4*30, 16] = [120, 16]
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True

        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # merge=True就是说4个扫描方向结果合起来，也不新增维度，就是同纬度拼起来
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def SSM_2D(self, if_double, x: torch.Tensor):  # 对应论文(c)2D-Selective Scan Module (2D-SSM) 这个就是状态空间模块，有这个就是mamba
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # [B, K, C(自定义通道数), L=H*W], K=4个扫描方向
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl [B, K=4, 新的通道数, L]
        # self.x_proj_weight [B, 新的通道数, L]
        # 新的通道数 = self.dt_rank + self.d_state * 2  就是压缩后的dts的维度+B与C两个状态参数的维度
        # dts实际上维度与输入的x是一致的都是[B, K, D=默认输入的通道数, H*W]
        # 这里的self.dt_rank实际上是对dts进行降维压缩（降维的倍数为self_d.state，后面会升维回来，目的是尽可能关注更精炼的隐空间信息
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        # torch.split的dim=2,就是沿着"新的通道数"，且成三段，每段的L一样，但是通道数不一样了
        # dts升维度回来
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        if if_double:
            xs = xs.double().view(B, -1, L)  # [B, K*C(自定义通道数), L=H*W], 4方向扫描的拼起来了！
            dts = dts.contiguous().double().view(B, -1, L)  # [B K D L] => [B, K*D, L], K=4个扫描方向, D=30是自定义输入的通道数
            Bs = Bs.double()
            Cs = Cs.double()
            Ds = self.Ds.double()
            dt_projs_bias = self.dt_projs_bias.double().view(-1)
        else:
            xs = xs.float().view(B, -1, L)  # [B, K*C(自定义通道数), L=H*W], 4方向扫描的拼起来了！
            dts = dts.contiguous().float().view(B, -1, L)  # [B K D L] => [B, K*D, L], K=4个扫描方向, D=30是自定义输入的通道数
            Bs = Bs.float()
            Cs = Cs.float()
            Ds = self.Ds.float()
            dt_projs_bias = self.dt_projs_bias.float().view(-1)

        As = -torch.exp(self.A_logs)

        out_y = self.selective_scan(
            if_double,
            xs, dts,
            As, Bs, Cs, Ds,  # B与C不一定非要变成[B, D, L]格式，因为self.selective_scan函数可以自适应
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == xs.dtype

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape
        xz = self.in_proj(x)  # 相对于(b)中分支并linear, 这里一步到位直接影身扩大2被维度，然后chunk分开2半相同大小的
        x, z = xz.chunk(2, dim=-1)  # chunk沿最后一维分割为2个变量
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # [B, C, H, W]
        y1, y2, y3, y4 = self.SSM_2D(self.if_double, x)  # [B, C, H*W]  SSM_2D中蕴含了展开的顺序的流程
        assert y1.dtype == x.dtype
        y = y1 + y2 + y3 + y4  # 这里居然没有加可学习的权重参数【可改进】
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)  # [B, H, W, C]  transpose调换两个维度的顺序
        y = self.out_norm(y)  # [B, H, W, C]
        y = y * F.silu(z)  # 两个分支是相点乘
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSM_V(nn.Module):  # 对应论文(b)Vision State Space Module (VSSM)
    def __init__(
            self,
            if_double,
            d_size_h,
            d_size_w,
            d_model,
            d_state=16,
            d_conv=3,  # 使用默认
            expand=2.,  # 就是模型导入的默认参数 mlp_ratio
            dt_rank="auto",  # 使用默认
            dt_min=0.001,  # 使用默认
            dt_max=0.1,  # 使用默认
            dt_init="random",  # 使用默认
            dt_scale=1.0,  # 使用默认
            dt_init_floor=1e-4,  # 使用默认
            dropout=0.,  # 使用默认
            conv_bias=True,  # 使用默认
            bias=False,  # 使用默认
            device=None,  # 使用默认
            dtype=None, ):  # 使用默认
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.if_double = if_double

        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.d_inner_hw = int(d_size_h * d_size_w)
        self.dt_rank = math.ceil((d_size_h * d_size_w) / 2) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs, )

        self.act = nn.SiLU()
        self.x_proj = (nn.Linear(self.d_inner_hw, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
                       nn.Linear(self.d_inner_hw, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),)
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner_hw, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner_hw, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),)
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.if_double, self.d_state, self.d_inner_hw, copies=2, merge=True)

        self.Ds = self.D_init(self.d_inner_hw, copies=2, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner_hw)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout > 0. else None

    @staticmethod
    # 全都是默认参数
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(if_double, d_state, d_inner, copies=1, device=None, merge=True):
        # merge=True就是说4个扫描方向结果合起来，也不新增维度，就是同纬度拼起来
        if if_double:
            dtype = torch.double
        else:
            dtype = torch.float

        A = repeat(
            torch.arange(1, d_state + 1, dtype=dtype, device=device),
            "n -> d n",  # n = d_state, d = d_inner
            d=d_inner,  # [d, n] = [30, 16]
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)  # [r, d, n] = [4, 30, 16]
            if merge:
                A_log = A_log.flatten(0, 1)  # [r*d, n] = [4*30, 16] = [120, 16]
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True

        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # merge=True就是说4个扫描方向结果合起来，也不新增维度，就是同纬度拼起来
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def SSM_2D(self, if_double, x: torch.Tensor):  # 对应论文(c)2D-Selective Scan Module (2D-SSM) 这个就是状态空间模块，有这个就是mamba
        B, H, W, L = x.shape  # L实际上是通道数，且被扩充了expand倍 = self.model * self.expand，在self.in_proj作用下
        D = H * W
        K = 2

        xs = torch.cat([x.view(B, D, L).unsqueeze(1), torch.flip(x.view(B, D, L), dims=[-1]).unsqueeze(1)], dim=1).view(
            B, 2, D, L)  # [B, K, D, L], K=2个扫描方向
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        if if_double:
            xs = xs.double().view(B, -1, L)  # [B, K*D(D=H*W), L=通道数], 4方向扫描的拼起来了！
            dts = dts.contiguous().double().view(B, -1, L)  # [B K D L] => [B, K*D(D=H*W), L=通道数], 4方向扫描的拼起来了！
            Bs = Bs.double()
            Cs = Cs.double()
            Ds = self.Ds.double()
            dt_projs_bias = self.dt_projs_bias.double().view(-1)
        else:
            xs = xs.view(B, -1, L)  # [B, K*D(D=H*W), L=通道数], 4方向扫描的拼起来了！
            dts = dts.contiguous().view(B, -1, L)  # [B, K*D(D=H*W), L=通道数], 4方向扫描的拼起来了！
            Bs = Bs
            Cs = Cs
            Ds = self.Ds
            dt_projs_bias = self.dt_projs_bias.view(-1)

        As = -torch.exp(self.A_logs)

        out_y = self.selective_scan(
            if_double,
            xs, dts,
            As, Bs, Cs, Ds,  # B与C不一定非要变成[B, D, L]格式，因为self.selective_scan函数可以自适应
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, 2, D, L)
        assert out_y.dtype == xs.dtype

        y1 = out_y[:, 0:1].squeeze(1)
        y2 = torch.flip(out_y[:, 1:2], dims=[-1]).squeeze(1)

        return y1, y2

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape
        xz = self.in_proj(x)  # 相对于(b)中分支并linear, 这里一步到位直接影身扩大2被维度，然后chunk分开2半相同大小的
        x, z = xz.chunk(2, dim=-1)  # chunk沿最后一维分割为2个变量
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W] 目的是适用于卷积操作
        x = self.act(self.conv2d(x)).permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        y1, y2 = self.SSM_2D(self.if_double, x)  # [B, H*W, C]  SSM_2D中蕴含了展开的顺序的流程
        assert y1.dtype == x.dtype
        y = y1 + y2  # [B, H*W, C] 展开
        y = torch.transpose(y, dim0=1, dim1=2).contiguous()  # [B, C, H*W]
        y = self.out_norm(y).view(B, -1, H, W).permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        y = y * F.silu(z)  # 两个分支是相点乘   # z [B, H, W, C]
        out = self.out_proj(y)  # 与self.in_proj(x)相对
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class RSSB_H(nn.Module):  # 对应论文(a) Residue State Space Block (RSSB)
    def __init__(
            self,
            if_double: bool = True,
            size_h: int = 0,
            size_w: int = 0,
            dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            mlp_ratio: float = 2.,
    ):
        super().__init__()
        self.size_h = size_h
        self.size_w = size_w
        self.ln1 = norm_layer(dim)  # 代码中所有dim都指的是通道数
        self.vssm = VSSM_H(if_double=if_double, d_model=dim, d_state=d_state, expand=mlp_ratio, dropout=attn_drop_rate)
        self.drop_path = DropPath(drop_path)
        self.scale = nn.Parameter(torch.ones(dim))
        self.ln2 = nn.LayerNorm(dim)
        self.convca = Conv_CA(dim)  # 就是RSSB中的最后Conv+CA两个模块，连在一起了
        self.scale2 = nn.Parameter(torch.ones(dim))

    def forward(self, input):
        B, L, C = input.shape  # x [B, H*W, C]
        input = input.view(B, self.size_h, self.size_w, C).contiguous()  # 由 [B, H*W, C] 恢复(解包)为 [B, H, W, C]
        # 此后都是4D张量
        x = self.ln1(input)  # 就是对每个H*W构成的平面进行归一化，一共有B*C个这样的平面  # [B, H, W, C]
        x = input * self.scale + self.drop_path(self.vssm(x))
        x = x * self.scale2 + self.convca(self.ln2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        # 此前都是4D张量
        x = x.view(B, -1, C).contiguous()  # x [B, H*W, C]
        return x


class RSSB_V(nn.Module):  # 对应论文(a) Residue State Space Block (RSSB)
    def __init__(
            # 除了下面三个float,其他都在float和double之间交替
            self,
            if_double: bool = True,
            size_h: int = 0,
            size_w: int = 0,
            dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            mlp_ratio: float = 2.,
    ):
        super().__init__()
        self.size_h = size_h
        self.size_w = size_w
        self.ln1 = norm_layer(size_h * size_w)  # 代码中所有dim都指的是通道数
        self.vssm = VSSM_V(if_double=if_double,
                           d_size_h=size_h, d_size_w=size_w, d_model=dim,
                           d_state=d_state, expand=mlp_ratio, dropout=attn_drop_rate)
        self.drop_path = DropPath(drop_path)
        self.scale = nn.Parameter(torch.ones(dim))
        self.ln2 = nn.LayerNorm(size_h * size_w)
        self.convca = Conv_CA(dim)  # 就是RSSB中的最后Conv+CA两个模块，连在一起了
        self.scale2 = nn.Parameter(torch.ones(dim))

    def forward(self, input):
        B, L, C = input.shape  # [B, H*W, C]
        input = input.permute(0, 2, 1).contiguous()  # [B, C, H*W]
        x = self.ln1(input)
        input = input.permute(0, 2, 1).view(B, self.size_h, self.size_w, C).contiguous()
        x = x.permute(0, 2, 1).view(B, self.size_h, self.size_w, C).contiguous()
        x = input * self.scale + self.drop_path(self.vssm(x))  # [B, H, W, C]
        x = x * self.scale2 + self.convca(
            self.ln2(x.view(B, -1, C).permute(0, 2, 1).contiguous())
            .view(B, C, self.size_h, self.size_w).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()  # x [B, H*W, C]
        return x


class Multi_RSSB(nn.Module):  # 对应论文的RSSG中多个(a)的顺序连接整合，不含Conv与残差连接
    def __init__(self,
                 if_double,
                 size_h,
                 size_w,
                 dim,
                 depth,
                 d_state,
                 mlp_ratio=2.,
                 drop_path=0.,
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint  # checkpointing 方式优化内存使用。尽管它可能导致计算时间的增加，但在显存受限的情况下，这种权衡通常是值得的。

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                RSSB_H(if_double=if_double,
                       size_h=size_h, size_w=size_w, dim=dim,
                       drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, d_state=d_state,
                       mlp_ratio=mlp_ratio))
            self.blocks.append(
                RSSB_V(if_double=if_double,
                       size_h=size_h, size_w=size_w, dim=dim,
                       drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, d_state=d_state,
                       mlp_ratio=mlp_ratio))

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


class RSSG(nn.Module):  # 对应论文的RSSG
    def __init__(self,
                 if_double,
                 size_h,
                 size_w,
                 dim,
                 depth,
                 d_state=16,
                 mlp_ratio=2.,
                 drop_path=0.,
                 use_checkpoint=False):
        super(RSSG, self).__init__()

        self.dim = dim
        self.multi_RSSB = Multi_RSSB(
            if_double=if_double,
            size_h=size_h,
            size_w=size_w,
            dim=dim,
            depth=depth,
            d_state=d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            use_checkpoint=use_checkpoint)

        # self.patch_embed = PatchEmbed(embed_dim=dim, norm_layer=None)
        # self.patch_unembed = PatchUnEmbed(size_h=size_h, size_w=size_w)

    def forward(self, x):
        # return self.patch_embed(self.patch_unembed(self.multi_RSSB(x))) + x
        return self.multi_RSSB(x) + x


# [展开]
class PatchEmbed(nn.Module):
    def __init__(self, embed_dim=96, norm_layer=None):  # embed_dim = C [通道数]
        super().__init__()
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # (B, C, H, W) => (B, C, H*W) => (B, H*W, C)
        if self.norm is not None:
            x = self.norm(x)
        return x


# [折叠]
class PatchUnEmbed(nn.Module):
    def __init__(self, size_h, size_w):
        super().__init__()
        self.size_h = size_h
        self.size_w = size_w

    def forward(self, x):
        x = x.transpose(1, 2).view(x.shape[0], x.shape[2], self.size_h, self.size_w)
        # (B, H*W, N) => (B, N, H*W) => (B, C, H, W)
        return x


def _init_weights(m):
    # [对线性层的处理]
    if isinstance(m, nn.Linear):
        # 权重使用截断正态分布初始化，偏置初始化为
        trunc_normal_(m.weight, std=.02)  # 截断正态分布的初始化方法
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    # [对层归一化层的处理]
    elif isinstance(m, nn.LayerNorm):
        # 偏置初始化为 0，权重初始化为 1.0
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH


def iwt_init(if_double, x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, :out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    if if_double:
        h = torch.zeros([out_batch, out_channel, out_height,
                         out_width]).double().to(x.device)
    else:
        h = torch.zeros([out_batch, out_channel, out_height,
                         out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self, if_double):
        super(IWT, self).__init__()
        self.requires_grad = False
        self.if_double = if_double

    def forward(self, x):
        return iwt_init(self.if_double, x)


class DeepFeatures(nn.Module):
    def __init__(self,
                 if_double,
                 size_h,
                 size_w,
                 embed_dim=180,
                 depths=(6, 6, 6, 6, 6, 6),
                 d_state=16,
                 mlp_ratio=2.,
                 drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False):
        super(DeepFeatures, self).__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio

        # Patch embedding layer
        self.patch_embed = PatchEmbed(embed_dim=embed_dim, norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate, inplace=True)

        # Create RSSG layers
        self.rssgs = nn.ModuleList()
        for i_layer in range(self.num_layers):
            rssg = RSSG(if_double=if_double,
                        size_h=size_h, size_w=size_w, dim=embed_dim,
                        depth=depths[i_layer], d_state=d_state, mlp_ratio=self.mlp_ratio,
                        use_checkpoint=use_checkpoint)
            self.rssgs.append(rssg)

        # 压缩每阶段通道数
        self.bottles = nn.ModuleList()
        for i_layer in range(self.num_layers):
            bottle = nn.Sequential(
                norm_layer(embed_dim * (i_layer + 2)),
                PatchUnEmbed(size_h=size_h, size_w=size_w),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim * (i_layer + 2), embed_dim, 3, 1, 1),
                PatchEmbed(embed_dim=embed_dim, norm_layer=None))
            self.bottles.append(bottle)

        self.patch_unembed = PatchUnEmbed(size_h=size_h, size_w=size_w)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        # RSSGs
        x_stage = x
        for rssg, bottle in zip(self.rssgs, self.bottles):
            x_rssg = rssg(x)
            x_stage = torch.cat([x_rssg, x_stage], dim=2)
            x = bottle(x_stage)
        x = self.patch_unembed(x)
        return x


class WaveletPyramid(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # 原卷积层替换为注意力引导的融合
        self.fuse_conv = nn.ModuleList([
            nn.Conv2d(embed_dim * 2, embed_dim, 3, 1, 1)
            for _ in range(3)])
        # 新增注意力权重生成层 输出权重尺寸为B,1,1,1, 也就是为每个融合结果计算求和时的权重
        self.attention = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(embed_dim * 2, embed_dim // 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim // 2, 1, 1),
                nn.Sigmoid()
            ) for _ in range(3)])

    def forward(self, LL, HL, LH, HH):
        fused = []
        for i, band in enumerate([HL, LH, HH]):
            # 拼接低频与高频特征
            cat_feat = torch.cat([LL, band], dim=1)  # [B, 2C, H, W]
            # 生成注意力权重
            attn = self.attention[i](cat_feat)  # [B,1,H,W]
            # 加权卷积融合
            weighted_feat = attn * self.fuse_conv[i](cat_feat)  # [B,C,H,W]
            fused.append(weighted_feat)
        return torch.stack(fused).mean(dim=0)


class Model(nn.Module):
    def __init__(self,
                 if_double,
                 in_chans=3,
                 out_chans=50,
                 size_h=10,
                 size_w=12,
                 embed_dim=180,
                 depths=(6, 6, 6, 6, 6, 6),
                 d_state=16,
                 mlp_ratio=2.,
                 drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False):
        super(Model, self).__init__()

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.shallow_feature = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.deep_features_LL = DeepFeatures(if_double=if_double,
                                             size_h=size_h, size_w=size_w, embed_dim=embed_dim,
                                             depths=depths, d_state=d_state, mlp_ratio=mlp_ratio,
                                             drop_rate=drop_rate, norm_layer=norm_layer, patch_norm=patch_norm,
                                             use_checkpoint=use_checkpoint)
        self.deep_features_HL = DeepFeatures(if_double=if_double,
                                             size_h=size_h, size_w=size_w, embed_dim=embed_dim,
                                             depths=depths, d_state=d_state, mlp_ratio=mlp_ratio,
                                             drop_rate=drop_rate, norm_layer=norm_layer, patch_norm=patch_norm,
                                             use_checkpoint=use_checkpoint)
        self.deep_features_LH = DeepFeatures(if_double=if_double,
                                             size_h=size_h, size_w=size_w, embed_dim=embed_dim,
                                             depths=depths, d_state=d_state, mlp_ratio=mlp_ratio,
                                             drop_rate=drop_rate, norm_layer=norm_layer, patch_norm=patch_norm,
                                             use_checkpoint=use_checkpoint)
        self.deep_features_HH = DeepFeatures(if_double=if_double,
                                             size_h=size_h, size_w=size_w, embed_dim=embed_dim,
                                             depths=depths, d_state=d_state, mlp_ratio=mlp_ratio,
                                             drop_rate=drop_rate, norm_layer=norm_layer, patch_norm=patch_norm,
                                             use_checkpoint=use_checkpoint)

        # ------------------------- reconstruction module ------------------------- #
        self.reconstruction_LL = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(embed_dim, out_chans, 1, 1, 0))
        self.reconstruction_HL = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(embed_dim, out_chans, 1, 1, 0))
        self.reconstruction_LH = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(embed_dim, out_chans, 1, 1, 0))
        self.reconstruction_HH = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(embed_dim, out_chans, 1, 1, 0))

        # ------------------------- weight initialization ------------------------- #
        self.apply(_init_weights)  # 将函数self._init_weights应用于当前模型的所有层

        # ------------------------- wavelet transform ----------------------------- #
        self.dwt = DWT()
        self.iwt = IWT(if_double)
        self.wavelet_pyramid = WaveletPyramid(embed_dim)

    def forward(self, x):
        # [Shallow feature extraction]
        x = self.shallow_feature(x)

        # [Deep feature extraction]
        x_LL, x_HL, x_LH, x_HH = self.dwt(x)
        x_LL = self.deep_features_LL(x_LL) + x_LL
        x_HL = self.deep_features_HL(x_HL) + x_HL
        x_LH = self.deep_features_LH(x_LH) + x_LH
        x_HH = self.deep_features_HH(x_HH) + x_HH

        # [Reconstruction]
        # [新增] 多尺度特征金字塔融合
        fused_LL = self.wavelet_pyramid(x_LL, x_HL, x_LH, x_HH) + x_LL  # 融合后的低频特征
        fused_HL = self.wavelet_pyramid(x_HL, x_LL, x_LH, x_HH) + x_HL  # 高频特征与低频交互
        fused_LH = self.wavelet_pyramid(x_LH, x_LL, x_HL, x_HH) + x_LH
        fused_HH = self.wavelet_pyramid(x_HH, x_LL, x_HL, x_LH) + x_HH

        # [Reconstruction] 使用融合后的特征
        x_LL = self.reconstruction_LL(fused_LL)
        x_HL = self.reconstruction_HL(fused_HL)
        x_LH = self.reconstruction_LH(fused_LH)
        x_HH = self.reconstruction_HH(fused_HH)

        # [合并]
        x_ALL = self.iwt(torch.cat([x_LL, x_HL, x_LH, x_HH], dim=1))
        return x_ALL


# [超参一体化]
def BuildModel(kwargs):
    if_double = kwargs.get('mambair', 3)['if_double']
    in_ch = kwargs.get('mambair', 3)['in_chans']
    out_ch = kwargs.get('mambair', 50)['out_chans']
    size_h = kwargs.get('mambair', 10)['size_h']
    size_w = kwargs.get('mambair', 12)['size_w']
    embed_dim = kwargs.get('mambair', 180)['embed_dim']
    depths = kwargs.get('mambair', (6, 6, 6, 6, 6, 6))['depths']
    d_state = kwargs.get('mambair', 16.)['d_state']
    mlp_ratio = kwargs.get('mambair', 2.)['mlp_ratio']
    drop_rate = kwargs.get('mambair', 0.)['drop_rate']
    norm_layer = kwargs.get('mambair', nn.LayerNorm)['norm_layer']
    patch_norm = kwargs.get('mambair', True)['patch_norm']
    use_checkpoint = kwargs.get('mambair', False)['use_checkpoint']

    return Model(if_double=if_double,
                 in_chans=in_ch,
                 out_chans=out_ch,
                 size_h=size_h,
                 size_w=size_w,
                 embed_dim=embed_dim,
                 depths=depths,
                 d_state=d_state,
                 mlp_ratio=mlp_ratio,
                 drop_rate=drop_rate,
                 norm_layer=norm_layer,
                 patch_norm=patch_norm,
                 use_checkpoint=use_checkpoint)
