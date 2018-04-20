import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.insert(0, '/models/')

import Attention
import Decoder

from encoder import Encoder


# vanilla seq2seq model
class NSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, embed_size, hidden_size, num_layers, dpt=0.2):
        super(NSeq2Seq, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, hidden_size, num_layers, dpt)
        # self.decoder = BasicDecoder(trg_vocab_size, embed_size, hidden_size, num_layers, dpt)
        # self.decoder = BasicAttentionDecoder(trg_vocab_size, embed_size, 2 * hidden_size, num_layers, dpt)
        # self.decoder = BasicBahdanauAttnDecoder(trg_vocab_size, embed_size, hidden_size, num_layers, dpt)
        # Store hyperparameters
        self.h_dim = hidden_size
        # self.vocab_size_trg, self.emb_dim_trg = embedding_trg.size()
        # self.bos_token = tokens_bos_eos_pad_unk[0]
        # self.eos_token = tokens_bos_eos_pad_unk[1]
        # self.pad_token = tokens_bos_eos_pad_unk[2]
        # self.unk_token = tokens_bos_eos_pad_unk[3]
        # self.reverse_input = reverse_input
        # Create encoder, decoder, attention
        # self.encoder = EncoderLSTM(embedding_src, h_dim, num_layers, dropout_p=dropout_p, bidirectional=bi)
        embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.decoder = Decoder.DecoderLSTM(embedding, hidden_size, num_layers * 2, dropout_p=dpt)
        self.attention = Attention.Attention(bidirectional=True, h_dim=self.h_dim)
        # Create linear layers to combine context and hidden state
        self.linear1 = nn.Linear(2 * self.h_dim, self.emb_dim_trg)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dpt)
        self.linear2 = nn.Linear(self.emb_dim_trg, self.vocab_size_trg)
        # Tie weights of decoder embedding and output
        # if True and self.decoder.embedding.weight.size() == self.linear2.weight.size():
        #     print('Weight tying!')
        self.linear2.weight = self.decoder.embedding.weight

    def forward(self, src, trg):
        if use_gpu: src = src.cuda()
        # Encode
        out_e, final_e = self.encoder(src)
        # Decode
        out_d, final_d = self.decoder(trg, final_e)
        # Attend
        context = self.attention(src, out_e, out_d)
        out_cat = torch.cat((out_d, context), dim=2)
        # Predict (returns probabilities)
        x = self.linear1(out_cat)
        x = self.dropout(self.tanh(x))
        x = self.linear2(x)
        x = F.log_softmax(x, dim = 2)
        return x

    def encode(self, src):
        return self.encoder(src)

    def generate(self, trg, src, enc_output, hidden=None):
        return self.decoder(trg, enc_output, hidden)

    def forward(self, src, trg):
        enc_output = self.encoder(src)
        output, hidden = self.decoder(trg, enc_output)
        return output, hidden