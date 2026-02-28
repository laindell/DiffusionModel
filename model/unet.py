import torch
import torch.nn as nn

from .timestep_embedding import SinusoidalPositionEmbeddings
from .blocks import ResidualBlock, AttentionBlock, Downsample, Upsample


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_multiplier: list = [1, 2, 4, 8],
        num_res_blocks: int = 2,
        time_embed_dim: int = 128,
        attention_resolutions: list = [8, 16],
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channel_multiplier = channel_multiplier
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = len(channel_multiplier)

        self.time_embed = SinusoidalPositionEmbeddings(time_embed_dim)
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.skip_channels = []

        channels = base_channels
        for level in range(self.num_resolutions):
            out_ch = base_channels * channel_multiplier[level]

            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ResidualBlock(
                        in_channels=channels,
                        out_channels=out_ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                    )
                )
                channels = out_ch
                self.skip_channels.append(channels)

            if level < self.num_resolutions - 1:
                self.downsamples.append(Downsample(channels))
                self.skip_channels.append(channels)

        mid_channels = base_channels * channel_multiplier[-1]
        self.middle_block1 = ResidualBlock(channels, mid_channels, time_embed_dim, dropout)
        self.middle_attention = AttentionBlock(mid_channels)
        self.middle_block2 = ResidualBlock(mid_channels, mid_channels, time_embed_dim, dropout)

        self.decoder_resblocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        channels = mid_channels
        skip_channels = list(self.skip_channels)

        for level in reversed(range(self.num_resolutions)):
            out_ch = base_channels * channel_multiplier[level]
            level_blocks = nn.ModuleList()

            # All decoder levels except the last (highest-resolution) consume
            # one extra skip produced by encoder downsampling.
            num_skip_for_level = num_res_blocks + (1 if level > 0 else 0)

            for _ in range(num_skip_for_level):
                skip_ch = skip_channels.pop()
                level_blocks.append(
                    ResidualBlock(
                        in_channels=channels + skip_ch,
                        out_channels=out_ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                    )
                )
                channels = out_ch

            self.decoder_resblocks.append(level_blocks)

            if level > 0:
                self.upsamples.append(Upsample(channels))

        self.output_conv = nn.Sequential(
            nn.GroupNorm(32, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_embed = self.time_embed(t)

        h = self.input_conv(x)
        skip_connections = []

        idx = 0
        for level in range(self.num_resolutions):
            for _ in range(self.num_res_blocks):
                h = self.encoder_blocks[idx](h, time_embed)
                skip_connections.append(h)
                idx += 1

            if level < self.num_resolutions - 1:
                h = self.downsamples[level](h)
                skip_connections.append(h)

        h = self.middle_block1(h, time_embed)
        h = self.middle_attention(h)
        h = self.middle_block2(h, time_embed)

        up_idx = 0
        for level_idx, level in enumerate(reversed(range(self.num_resolutions))):
            for block in self.decoder_resblocks[level_idx]:
                skip = skip_connections.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, time_embed)

            if level > 0:
                h = self.upsamples[up_idx](h)
                up_idx += 1

        h = self.output_conv(h)
        return h

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_unet(
    image_size: int = 32,
    in_channels: int = 3,
    out_channels: int = 3,
    base_channels: int = 64,
    channel_multiplier: list = [1, 2, 4, 8],
    num_res_blocks: int = 2,
    time_embed_dim: int = 128,
    attention_resolutions: list = [8, 16],
    dropout: float = 0.1,
) -> UNet:
    return UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        channel_multiplier=channel_multiplier,
        num_res_blocks=num_res_blocks,
        time_embed_dim=time_embed_dim,
        attention_resolutions=attention_resolutions,
        dropout=dropout,
    )
