import torch
import torch.nn as nn
from torch.autograd import Variable
from .layer_util import *
import numpy as np
import functools

class GlobalTwoStreamGenerator(nn.Module):
    def __init__(self, input_nc, output_nc=3, ngf=64, z_dim=16, n_downsampling=3, n_blocks=9, norm_layer='instance',
                 padding_type='reflect', use_skip=False, which_stream='ctx', use_output_gate=True,
                 feat_fusion='early_add', extra_embed=False):
        assert(n_blocks >= 0)
        assert( not (not ('label' in which_stream) and ('late' in feat_fusion)))
        assert( not (not ('ctx' in which_stream) and ('late' in feat_fusion)))
        super(GlobalTwoStreamGenerator, self).__init__()
        self.norm_layer = get_norm_layer(norm_layer)
        self.ngf = ngf
        self.z_dim=z_dim
        self.n_downsampling = n_downsampling
        self.padding_type=padding_type
        activation = nn.ReLU(True)
        self.activation = activation
        self.output_nc = output_nc
        self.n_blocks = n_blocks
        self.use_skip = use_skip
        self.which_stream = which_stream
        self.use_output_gate=use_output_gate
        self.feat_fusion = feat_fusion
        feat_dim = self.ngf*2**n_downsampling
        self.feat_dim = feat_dim
        self.extra_embed = extra_embed

        ctx_dim = 3 if not extra_embed else 6
        self.ctx_inputEmbedder = self.get_input(ctx_dim)
        self.ctx_downsampler = self.get_downsampler()
        self.noise_fuser = self.get_fuse_layer()
        self.latent_embedder = self.get_embedder(feat_dim, n_blocks)
        self.decoder = self.get_upsampler()
        self.outputEmbedder = self.get_output()

    def get_input(self, input_nc):
        model = [nn.ReflectionPad2d(3),
                nn.Conv2d(input_nc, self.ngf, kernel_size=7, padding=0),
                self.norm_layer(self.ngf),
                self.activation]
        return nn.Sequential(*model)

    def get_downsampler(self):
        ### downsample
        model = []
        for i in range(self.n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(self.ngf * mult, self.ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      self.norm_layer(self.ngf * mult * 2),
                      self.activation]
        return nn.Sequential(*model)

    def get_embedder(self, feat_dim, n_blocks):
        ### resnet blocks
        model = []
        for i in range(n_blocks):
            model += [ResnetBlock(feat_dim,
                padding_type=self.padding_type,
                activation=self.activation,
                norm_layer=self.norm_layer)]
        return nn.Sequential(*model)

    def get_upsampler(self):
        ### upsample
        model = []
        for i in range(self.n_downsampling):
            mult = 2**(self.n_downsampling - i)
            dim_in = self.ngf * mult
            dim_out = int(self.ngf * mult / 2)
            if self.use_skip and i > 0:
                dim_in = dim_in*2
            model += [nn.ConvTranspose2d(dim_in, dim_out,
                            kernel_size=3, stride=2, padding=1, output_padding=1),
                       self.norm_layer(int(self.ngf * mult / 2)),
                       self.activation]
        return nn.Sequential(*model)

    def get_output(self):
        model = [nn.ReflectionPad2d(3),
                nn.Conv2d(self.ngf, self.output_nc, kernel_size=7, padding=0),
                nn.Tanh()]
        return nn.Sequential(*model)

    def get_fuse_layer(self):
        model = [nn.Conv2d(self.feat_dim +self.z_dim, self.feat_dim, kernel_size=3, padding=1),
                self.norm_layer(self.feat_dim),
                self.activation]
        return nn.Sequential(*model)


    def forward_encoder(self, inputEmbedder, encoder, input, use_skip):
        enc_feats = []
        enc_feat = inputEmbedder(input)
        for i, layer in enumerate(encoder):
            enc_feat = layer(enc_feat)
            if use_skip and ((i < self.n_downsampling*3-1) and (i % 3 == 2)): # super-duper hard-coded
                enc_feats.append(enc_feat)
        return enc_feat, enc_feats

    def forward_embedder(self, ctx_feat):
        embed_feat = self.latent_embedder(ctx_feat)
        return embed_feat

    def forward_decoder(self, decoder, outputEmbedder, dec_feat, enc_feats):
        for i, layer in enumerate(decoder):
            if (self.use_skip and len(enc_feats) > 0) and ((i > 0) and (i % 3 ==0)): # super-duper hard-coded
                dec_feat = torch.cat((enc_feats[-int((i-3)/3)-1], dec_feat),1)
            dec_feat = layer(dec_feat)
        output = outputEmbedder(dec_feat)
        return output

    def forward(self, img, noise, mask):
        ctx_feat, ctx_feats = self.forward_encoder(self.ctx_inputEmbedder, self.ctx_downsampler, img, self.use_skip)
        # fuse the noise with feature
        noise = noise.expand(noise.size(0),noise.size(1),ctx_feat.size(2), ctx_feat.size(3))
        combined_feat = torch.cat((ctx_feat, noise),dim=1)
        combined_feat = self.noise_fuser(combined_feat)
        # do embedding
        embed_feat = self.forward_embedder(combined_feat)
        output = self.forward_decoder(self.decoder, self.outputEmbedder, embed_feat, ctx_feats)
        if self.use_output_gate:
            mask_output = mask.repeat(1, self.output_nc, 1, 1)
            #output = (1-mask_output)*img + mask_output*output
            output = (1-mask_output)*img[:,:3,:,:] + mask_output*output

        return output

