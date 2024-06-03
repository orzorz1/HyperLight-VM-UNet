from nnunet.network_architecture.neural_network import SegmentationNetwork as SN
from monai.networks.nets.vit import ViT
from torch import nn
model_patch_size = [32,128,128]
model_batch_size = 20
model_num_pool_op_kernel_sizes = [[2, 2]]
class custom_net(SN):

    def __init__(self, num_classes):
        super(custom_net, self).__init__()
        self.params = {'content': None}
        self.conv_op = nn.Conv3d
        self.do_ds = False
        self.num_classes = num_classes
        
		######## self.model 设置自定义网络 by Sleeep ########
        self.model = UltraLight_VM_UNet(num_classes=num_classes, input_channels=1, c_list=[8,16,24,32,48,64],
                split_att='fc', bridge=True)
        ######## self.model 设置自定义网络 by Sleeep ########
        
        self.name = "UltraLight_VM_UNet"

    def forward(self, x):
        out = self.model(x)
        if self.do_ds:
            return [out, ]
        else:
            return out


def create_model():

    return custom_net(num_classes=2)

import torch
from torch import nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
import math
from mamba_ssm import Mamba


class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # LayerNorm is typically applied across each feature, hence we keep it unchanged
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim // 4,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()  # This computes the product of all spatial dimensions
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


class DimensionalAttention(nn.Module):
    def __init__(self, dim):
        super(DimensionalAttention, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5  # 缩放因子

    def forward(self, x):
        # x: (batch, channel, Dim)
        batch_size, num_channels, dim = x.size()
        
        # 计算query, key, value
        query = self.query(x)  # (batch, channel, Dim)
        key = self.key(x)      # (batch, channel, Dim)
        value = self.value(x)  # (batch, channel, Dim)

        # 计算注意力分数
        attention_scores = torch.bmm(query, key.transpose(1, 2))  # (batch, channel, channel)
        attention_scores = attention_scores * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch, channel, channel)

        # 应用注意力权重
        attention_output = torch.bmm(attention_weights, value)  # (batch, channel, Dim)
        return attention_output


class AxialAttentionModule(nn.Module):
    def __init__(self, channels, in_shape):
        super().__init__()
        self.pool_w = nn.AdaptiveMaxPool3d((None, 1, 1))
        self.pool_h = nn.AdaptiveMaxPool3d((1, None, 1))
        self.pool_d = nn.AdaptiveMaxPool3d((1, 1, None))
        
        self.att_w = DimensionalAttention(in_shape[0])
        self.att_h = DimensionalAttention(in_shape[1])
        self.att_d = DimensionalAttention(in_shape[2])

    def forward(self, x):
        # x shape: (B, C, W, H, D)
        w = self.pool_w(x).squeeze(-1).squeeze(-1)  # (B, C, W)
        h = self.pool_h(x).squeeze(-1).squeeze(-2)  # (B, C, H)
        d = self.pool_d(x).squeeze(-2).squeeze(-2)  # (B, C, D)
        
        attn_w = self.att_w(w)  # (B, C, W)
        attn_h = self.att_h(h)  # (B, C, H)
        attn_d = self.att_d(d)  # (B, C, D)
        
        attn_w = attn_w / attn_w.max(dim=-1, keepdim=True)[0]
        attn_h = attn_h / attn_h.max(dim=-1, keepdim=True)[0]
        attn_d = attn_d / attn_d.max(dim=-1, keepdim=True)[0]

        # Outer product to reconstruct the (W, H, D) attention
        attn_wh = attn_w.unsqueeze(-2) * attn_h.unsqueeze(-1)  # (B, C, W, H)
        attn_whd = attn_wh.unsqueeze(-1) * attn_d.unsqueeze(-2).unsqueeze(-2)  # (B, C, W, H, D)
        
        # Ensure output shape matches the input shape
        attn_whd = F.interpolate(attn_whd, size=x.shape[2:], mode='trilinear', align_corners=False)
        
        return attn_whd



class SharedAttentionModule(nn.Module):
    def __init__(self):
        super(SharedAttentionModule, self).__init__()
        self.attention = AxialAttentionModule(48, (1, 4, 4))

    def forward(self, inputs):
        outputs = []
        for x in inputs:
            original_shape = x.shape[2:]
            x_resized = F.interpolate(x, size=(1, 4, 4), mode='trilinear', align_corners=False)
            attn_output = self.attention(x_resized)
            output = F.interpolate(attn_output, size=original_shape, mode='trilinear', align_corners=False)
            outputs.append(output)
        return outputs


class Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()

        self.satt = SharedAttentionModule()

    def forward(self, t1, t2, t3, t4, t5):
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5
        out = self.satt([t1, t2, t3, t4, t5])
        satt1, satt2, satt3, satt4, satt5 = out[0], out[1], out[2], out[3], out[4]
        t1, t2, t3, t4, t5 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4, satt5 * t5

        t1, t2, t3, t4, t5 = t1 + r1, t2 + r2, t3 + r3, t4 + r4, t5 + r5
        return t1, t2, t3, t4, t5


class UltraLight_VM_UNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc', bridge=True):
        super().__init__()
        self.bridge = bridge

        self.encoder1 = nn.Sequential(nn.Conv3d(input_channels, c_list[0], 3, stride=1, padding=1))
        self.encoder2 = nn.Sequential(nn.Conv3d(c_list[0], c_list[1], 3, stride=1, padding=1))
        self.encoder3 = nn.Sequential(nn.Conv3d(c_list[1], c_list[2], 3, stride=1, padding=1))
        self.encoder4 = nn.Sequential(PVMLayer(input_dim=c_list[2], output_dim=c_list[3]))
        self.encoder5 = nn.Sequential(PVMLayer(input_dim=c_list[3], output_dim=c_list[4]))
        self.encoder6 = nn.Sequential(PVMLayer(input_dim=c_list[4], output_dim=c_list[5]))

        if bridge:
            self.scab = Att_Bridge(c_list, split_att)
            print('SC_Att_Bridge was used')

        self.decoder1 = nn.Sequential(PVMLayer(input_dim=c_list[5], output_dim=c_list[4]))
        self.decoder2 = nn.Sequential(PVMLayer(input_dim=c_list[4], output_dim=c_list[3]))
        self.decoder3 = nn.Sequential(PVMLayer(input_dim=c_list[3], output_dim=c_list[2]))
        self.decoder4 = nn.Sequential(nn.Conv3d(c_list[2], c_list[1], 3, stride=1, padding=1))
        self.decoder5 = nn.Sequential(nn.Conv3d(c_list[1], c_list[0], 3, stride=1, padding=1))

        self.final = nn.Conv3d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        out = F.gelu(F.max_pool3d(self.encoder1(x), 2, 2))
        t1 = out

        out = F.gelu(F.max_pool3d(self.encoder2(out), 2, 2))
        t2 = out

        out = F.gelu(F.max_pool3d(self.encoder3(out), 2, 2))
        t3 = out

        out = F.gelu(F.max_pool3d(self.encoder4(out), 2, 2))
        t4 = out

        out = F.gelu(F.max_pool3d(self.encoder5(out), 2, 2))
        t5 = out


        if self.bridge:
            t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)


        out = F.gelu(self.encoder6(out))

        out = F.gelu(self.decoder1(out))
        out = out + t5

        out = F.gelu(F.interpolate(self.decoder2(out), scale_factor=2, mode='trilinear', align_corners=True))
        out = out + t4

        out = F.gelu(F.interpolate(self.decoder3(out), scale_factor=2, mode='trilinear', align_corners=True))
        out = out + t3

        out = F.gelu(F.interpolate(self.decoder4(out), scale_factor=2, mode='trilinear', align_corners=True))
        out = out + t2

        out = F.gelu(F.interpolate(self.decoder5(out), scale_factor=2, mode='trilinear', align_corners=True))
        out = out + t1

        out = F.interpolate(self.final(out), scale_factor=2, mode='trilinear', align_corners=True)

        return torch.sigmoid(out)



import torch
from thop import profile
import torch.nn.functional as F

def test_model():
    # 创建模型
    model = create_model().cuda()

    input_data = torch.randn(1, 1, 32,128, 128).cuda()  # Batch size 1, 2 channels, 64x128x128 volume

    # 计算 FLOPS 和参数数量
    flops, params = profile(model, inputs=(input_data,))
    print(f'FLOPS: {flops}, Parameters: {params}')

    # 测试显存占用
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        output = model(input_data)
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # 转换为 MB

    print(f'Peak memory usage: {peak_memory:.2f} MB')

# 运行测试函数
if __name__ == "__main__":
    test_model()
