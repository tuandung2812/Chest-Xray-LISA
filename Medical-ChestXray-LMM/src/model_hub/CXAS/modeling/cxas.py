from torch import nn
from torch.nn import functional as F


class CXAS(nn.Module):
    def __init__(
            self,
            image_encoder,
            image_decoder,
            dropout=0
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.image_decoder = image_decoder
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, data):
        backbone_dict = self.image_encoder(data)

        down1 = backbone_dict['feats_1_map']
        down2 = backbone_dict['feats_2_map']
        down3 = backbone_dict['feats_3_map']
        down4 = backbone_dict['feats_4_map']
        down5 = backbone_dict['feats_last_map']

        up1    = self.dropout(self.image_decoder.up1(down5, down4))
        up2    = self.dropout(self.image_decoder.up2(up1, down3))
        up3    = self.dropout(self.image_decoder.up3(up2, down2))
        up4    = self.dropout(self.image_decoder.up4(up3, down1))
        
        up     = F.interpolate(up4, data.shape[2:], mode='bilinear')
        logits = self.image_decoder.out(up)

        output = {
            "feat": up4,
            "logits": logits
        }

        return output
