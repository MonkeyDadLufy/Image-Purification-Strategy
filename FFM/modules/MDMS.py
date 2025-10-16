

import math
import torch
import torch.nn as nn

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    # assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2  
   
    emb = math.log(10000) / (half_dim - 1) 
    
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)  
    
    emb = emb.to(device=timesteps.device) 
  
    emb = timesteps.float()[:, None] * emb[None, :]  
    
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  

    if embedding_dim % 2 != 0:  
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))  
    
    return emb  
   


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=8, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class HighMixer(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1,
                 **kwargs, ):
        super().__init__()

        self.cnn_in = cnn_in = dim
        # self.pool_in = pool_in = dim-cnn_in

        self.cnn_dim = cnn_dim = cnn_in
        # self.pool_dim = pool_dim = pool_in

        self.conv1 = nn.Conv2d(cnn_in, cnn_dim, kernel_size=1, stride=1, padding=0, bias=False)
        # # self.conv1 = nn.Conv2d(cnn_in, cnn_dim, kernel_size=1, stride=1, padding=0, bias=False, groups=4)
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(cnn_in, cnn_in, kernel_size=1, stride=1, padding=0, bias=False, groups=cnn_in),
        #     nn.Conv2d(cnn_in, cnn_dim, kernel_size=1, stride=1, padding=0, bias=False)
        # )
        self.proj1 = nn.Conv2d(cnn_dim, cnn_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False,
                               groups=cnn_dim)
        self.mid_gelu1 = nn.GELU()



    def forward(self, x):
        # B, C H, W

        cx = x[:, :self.cnn_in, :, :].contiguous()
        cx = self.conv1(cx)
        cx = self.proj1(cx)
        cx = self.mid_gelu1(cx)

        return cx


class LowMixer(nn.Module):
    def __init__(self, dim, pool_size=2):
        super().__init__()
        self.pool_size = pool_size
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=dim, eps=1e-6, affine=True)
        self.act = nn.GELU()

        self.pool = nn.AvgPool2d(pool_size, stride=pool_size, padding=0, count_include_pad=False) if pool_size > 1 else nn.Identity()
        self.uppool = nn.Upsample(scale_factor=pool_size) if pool_size > 1 else nn.Identity()

    def forward(self, x):
        # Pooling
        x = self.pool(x)

        # Convolution
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        # Upsampling
        x = self.uppool(x)

        return x


class Mixer(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., attention_head=1, pool_size=2):
        super().__init__()
        self.num_heads = num_heads
        self.low_dim = low_dim = dim // 2
        self.high_dim = high_dim = dim - low_dim
        self.high_mixer = HighMixer(high_dim)
        self.low_mixer = LowMixer(low_dim, pool_size=pool_size)

        self.conv_fuse = nn.Conv2d(low_dim + high_dim, low_dim + high_dim, kernel_size=3, stride=1, padding=1, bias=False, groups=low_dim + high_dim)
        self.proj = nn.Conv2d(low_dim + high_dim, dim, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)

        self.freblock = FreBlock(dim, dim)
        self.finalproj = nn.Conv2d(2 * dim, dim, 1, 1, 0)

    def forward(self, x):
        B, C, H, W = x.shape

        # High-frequency mixing
        hx = x[:, :self.high_dim, :, :].contiguous()
        hx = self.high_mixer(hx)

        # Low-frequency mixing
        lx = x[:, self.high_dim:, :, :].contiguous()
        lx = self.low_mixer(lx)

        # Concatenate and fuse
        x = torch.cat((hx, lx), dim=1)
        x = x + self.conv_fuse(x)
        x_spatial = self.proj(x)

        # Frequency block
        x_freq = self.freblock(x)

        # Final projection
        x_out = torch.cat((x_spatial, x_freq), 1)
        x_out = self.finalproj(x_out)
        x_out = self.proj_drop(x_out)

        return x_out + x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512, incep=False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        if incep:
            self.conv2 = Mixer(dim=out_channels)
        else:
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_

class FreBlock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(FreBlock, self).__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,1,1,0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(in_channels,out_channels,1,1,0))
        self.processpha = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0))

    def forward(self,x):
        xori = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.processmag(mag)
        pha = self.processpha(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out1 = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out1
class DiffusionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        ch, out_ch, ch_mult = 64, 3, (1,2,4,8)
        num_res_blocks = 1
        attn_resolutions = []
        dropout = 0.1
        in_channels = 3
        resolution = 512
        resamp_with_conv = True

        self.ch = ch                #128
        self.temb_ch = self.ch*4    #512
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.inceplayers=2

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                if i_level+self.inceplayers<self.num_resolutions:
                    block.append(ResnetBlock(in_channels=block_in,
                                             out_channels=block_out,
                                             temb_channels=self.temb_ch,
                                             dropout=dropout,incep=True))
                else:
                    block.append(ResnetBlock(in_channels=block_in,
                                             out_channels=block_out,
                                             temb_channels=self.temb_ch,
                                             dropout=dropout,incep=True))
                block_in = block_out
                # if curr_res in attn_resolutions:
                #     attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1 :
                if ch_mult[i_level]*2==ch_mult[i_level+1]:
                    down.downsample = Downsample(block_in, resamp_with_conv)
                    curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,incep=True)
        # self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,incep=True)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                if i_level+self.inceplayers>=self.num_resolutions:
                    block.append(ResnetBlock(in_channels=block_in+skip_in,
                                             out_channels=block_out,
                                             temb_channels=self.temb_ch,
                                             dropout=dropout,incep=True))
                else:
                    block.append(ResnetBlock(in_channels=block_in+skip_in,
                                             out_channels=block_out,
                                             temb_channels=self.temb_ch,
                                             dropout=dropout,incep=True))
                block_in = block_out
                # if curr_res in attn_resolutions:
                #     attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0 :
                if ch_mult[i_level]== ch_mult[i_level-1]*2:
                    up.upsample = Upsample(block_in, resamp_with_conv)
                    curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t):
        assert x.shape[2] == x.shape[3]
        # timestep embedding
        temb1 = get_timestep_embedding(t, self.ch)  # 修改这里

        temb = self.temb.dense[0](temb1)  # (1, 512)
        temb = nonlinearity(temb)  # (1, 512)
        temb = self.temb.dense[1](temb)  # (1, 512)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

if __name__ == '__main__':
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    vf = DiffusionUNet().to(device)
    print(f"模型参数量: {round(sum(p.numel() for p in vf.parameters() if p.requires_grad) / 1_000_000, 2)}M")
    input = torch.randn(1, 3, 512, 512).to(device)
    output = vf(input, torch.tensor([0.5], device=device))
    print(output.size())
