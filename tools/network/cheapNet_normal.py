import torch.nn as nn
import torch


def encoder_block(in_channels, out_channels, slope):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, (3, 3), stride=(2, 2), padding=(1, 1)),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(slope)
    )
    return block


def decoder_block(in_channels, out_channels, slope):
    block = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(slope)
    )
    return block


class CheapNetNormal(nn.Module):
    def __init__(self, groups, slope=0.2):
        super(CheapNetNormal, self).__init__()

        # [b,1,512,512] ==> [b,groups,512,512]
        self.encoder1 = nn.Sequential(
            nn.Conv2d(1, groups, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(groups),
            nn.LeakyReLU(slope)
        )

        # [b,groups,512,512] ==> [b,groups,256,256]
        self.encoder2 = encoder_block(groups, groups, slope)

        # [b,groups,256,256] ==> [b,groups,128,128]
        self.encoder3 = encoder_block(groups, groups, slope)

        # [b,groups,128,128] ==> [b,groups,64,64]
        self.encoder4 = encoder_block(groups, groups, slope)

        # [b,groups,64,64] ==> [b,groups,32,32]
        self.bottleneck = encoder_block(groups, groups, slope)

        # [b,groups,32,32] ==> [b,groups,64,64]
        self.decoder1 = decoder_block(groups, groups, slope)

        # [b,groups,64,64] ==> [b,groups,128,128]
        self.decoder2 = decoder_block(groups*2, groups, slope)

        # [b,groups,128,128] ==> [b,groups,256,256]
        self.decoder3 = decoder_block(groups*2, groups, slope)

        # [b,groups,256,256] ==> [b,groups,512,512]
        self.decoder4 = decoder_block(groups*2, groups, slope)

        # [b,groups,512,512] ==> [b,1,512,512]
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=groups*2, out_channels=1, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(slope),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoder1 = self.encoder1(x)      # [b,1,512,512] ==> [b,g,512,512]
        encoder2 = self.encoder2(encoder1)  # [b,g,512,512] ==> [b,g,256,256]
        encoder3 = self.encoder3(encoder2)    # [b,g,256,256] ==> [b,g,128,128]
        encoder4 = self.encoder4(encoder3)    # [b,g,128,128] ==> [b,g,64,64]

        bottleneck = self.bottleneck(encoder4)    # [b,g,64,64] ==> [b,g,32,32]

        decoder1 = self.decoder1(bottleneck)  # [b,g,32,32] ==> [b,g,64,64]
        decoder2 = self.decoder2(torch.cat([encoder4, decoder1], 1))      # [b,g,64,64]+[b,g,64,64] ==> [b,g,128,128]
        decoder3 = self.decoder3(torch.cat([encoder3, decoder2], 1))      # [b,g,128,128]+[b,g,128,128] ==> [b,g,256,256]
        decoder4 = self.decoder4(torch.cat([encoder2, decoder3], 1))  # [b,g,256,256]+[b,g,256,256] ==> [b,g,512,512]

        output = self.output(torch.cat([encoder1, decoder4], 1))      # [b,g,512,512] ==> [b,1,512,512]
        return output


if __name__ == "__main__":
    from torchinfo import summary
    group = 1024
    model = CheapNetNormal(groups=group)

    # 打印网络结构图
    summary(model, input_size=(1, 1, 512, 512), device="cpu",
            col_names=["input_size", "output_size", "num_params", 'mult_adds'])

    # 计算参数
    from thop import profile
    input = torch.randn(1, 1, 512, 512)
    flops, parms = profile(model, inputs=(input, ))
    print(f"Group:{group} FLOPs:{flops/1e9}G,params:{parms/1e6}M")

    # img = torch.randn(1, 1, 512, 512)
    # out = model(img)
    # print(out)

    # predict = out > 0.5
    # pre = predict.type(torch.int8).numpy()
    # print(pre)







