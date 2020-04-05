from fairseq.models.roberta import XLMRModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class XLMR_Encoder(nn.Module):

    def __init__(self, pretrained_path, hidden_size, dropout_p, device='cuda'):
        super().__init__()

        
        self.xlmr = XLMRModel.from_pretrained(pretrained_path)
        self.model = self.xlmr.model
        self.dropout = nn.Dropout(dropout_p)
        
        self.device=device

    def forward(self, src_tensor):
        #print(src_tensor)
        transformer_out, _ = self.model(src_tensor)#, features_only=True)

        return transformer_out

    def encode_word(self, s):
        """
        takes a string and returns a list of token ids
        """
        tensor_ids = self.xlmr.encode(s)
        # remove <s> and </s> ids
        return tensor_ids.cpu().numpy().tolist()[1:-1]
