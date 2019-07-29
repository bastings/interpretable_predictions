from torch import nn
from latent_rationale.common.util import get_encoder


class Encoder(nn.Module):
    """
    This is a general component that encodes a sentence,
    and nothing more.

    For what is called the "Encoder" in Lei et al., see "Classifier".

    This class supports LSTM and RCNN layers.

    If a z is provided as well as a mask, time steps where z==0 are skipped.

    """

    def __init__(self,
                 embed:        nn.Embedding = None,
                 hidden_size:  int = 200,
                 dropout:      float = 0.1,
                 layer:        str = "rcnn",
                 ):

        super(Encoder, self).__init__()

        self.embed_layer = nn.Sequential(
            embed,
            nn.Dropout(p=dropout)
        )

        emb_size = embed.weight.shape[1]
        self.enc_size = hidden_size * 2
        self.enc_layer = get_encoder(layer, emb_size, hidden_size)

    def forward(self, x, mask, z=None):

        rnn_mask = mask
        emb = self.embed_layer(x)

        # apply z to main inputs (if provided), and mask the mask with z
        if z is not None:
            z_mask = mask.unsqueeze(-1).float() * z.squeeze(1)  # [B, T, 1]
            rnn_mask = z_mask.squeeze(-1) > 0.  # z could be continuous
            emb = emb * z_mask

        # z is also used to control when the encoder layer is active
        lengths = mask.long().sum(1)

        # encode the sentence
        outputs, final = self.enc_layer(emb, rnn_mask, lengths)

        return outputs, final
