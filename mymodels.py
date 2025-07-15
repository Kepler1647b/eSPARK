import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange, reduce
import sys
from models.aggregators.model_utils import FeedForward, PreNorm
from multihead_diffattn import MultiheadDiffAttn


class Attention(nn.Module):
    def __init__(self, dim=512, heads=8, dim_head=512 // 8, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, register_hook=False):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        # save self-attention maps
        self.save_attention_map(attn)
        if register_hook:
            attn.register_hook(self.save_attn_gradients)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def get_self_attention(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        return attn


class CrossAttention(nn.Module):
    def __init__(self, dim=512, heads=8, dim_head=512 // 8, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim), nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, context, mask=None):
        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h=self.heads)
        k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), self.to_kv(context).chunk(2, dim=-1)
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            dots.masked_fill_(mask, float('-inf'))

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class GLU(nn.Module):
    def __init__(self, input_dim, output_dim, activation = nn.SiLU()):
        super(GLU, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)
        self.fc_gate = nn.Linear(input_dim, input_dim)
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.activation = activation

    def forward(self, x):
        x_tr = self.fc(x)
        gate = self.fc_gate(x)
        # x = self.activation(x) * torch.sigmoid(gate)
        x = self.activation(gate) * x_tr
        x = self.fc_out(x)
        return x


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class CLAM_SB(nn.Module):
    def __init__(self, size_arg = "small", dropout = False, n_classes=2):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [512, 256, 128], "big": [256, 192, 128]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(dropout))
        attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        self.n_classes = n_classes

        # initialize_weights(self)

    def forward(self, h):
        # print(h.shape)
        h = h.squeeze()
        h = h.float()
        A, h = self.attention_net(h)  # NxK
        # print(A.shape, h.shape)
        A = torch.transpose(A, 1, 0)  # KxN

        A = F.softmax(A, dim=1)  # softmax over N

        # print(A.shape, h.shape)

        M = torch.mm(A, h)
        logits = self.classifiers(M)

        return logits


class batch_CLAM_SB(nn.Module):
    def __init__(self, size_arg = "small", dropout = False, n_classes=2):
        super(batch_CLAM_SB, self).__init__()
        self.size_dict = {"small": [512, 256, 128], "big": [256, 192, 128]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(dropout))
        attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        self.n_classes = n_classes

        # initialize_weights(self)

    def forward(self, h):
        # print(h.shape)
        h = h.squeeze()
        h = h.float()
        A, h = self.attention_net(h)  # NxK
        # print(A.shape, h.shape)
        A = torch.transpose(A, 1, 2)  # KxN

        A = F.softmax(A, dim=-1)  # softmax over N

        # print(A.shape, h.shape)

        M = torch.bmm(A, h)
        # print(M.shape)
        M = M.squeeze(1)
        logits = self.classifiers(M)

        return logits


class TransformerBlocks(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
                    ]
                )
            )

    def forward(self, x, register_hook=False):
        for attn, ff in self.layers:
            x = attn(x, register_hook=register_hook) + x
            x = ff(x) + x
        return x


class TransformerBlocksCross(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        # PreNorm(
                        #     dim, CrossAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
                        # ),
                        # PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)
                        # )
                        nn.LayerNorm(dim),
                        CrossAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        nn.LayerNorm(dim),
                        FeedForward(dim, mlp_dim, dropout=dropout)
                    ]
                )
            )

    def forward(self, x, context, mask=None):
        for ln1, attn, ln2, ff in self.layers:
            # x = attn(x, context, mask=mask) + x
            # x = ff(x) + x
            x = ln1(x)
            x = attn(x, context, mask=mask) + x
            x = ln2(x)
            x = ff(x) + x
        return x
    

class DiffTransformerBlocks(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim, MultiheadDiffAttn(embed_dim=dim, depth=i, num_heads=heads)
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)
                        )
                    ]
                )
            )

    def forward(self, x, register_hook=False):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    

class CLAM_SB_multimodal(nn.Module):
    def __init__(self, size_arg = "small", dropout = False, n_classes=2, ct_dim=99, fusion_dim=128):
        super(CLAM_SB_multimodal, self).__init__()
        self.size_dict = {"small": [512, 256, 128], "big": [256, 192, 128]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(dropout))
        attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], fusion_dim)
        self.n_classes = n_classes

        # initialize_weights(self)
        # ct modalities
        self.preprocess_ct = nn.Linear(ct_dim, fusion_dim)

        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(ct_dim, 128))
        for _ in range(3- 1):
            self.layers.append(nn.Linear(128, 128))
            self.layers.append(eval("nn." + 'Sigmoid')())
            self.layers.append(nn.Dropout(dropout))

        self.multi_modal = nn.Linear(fusion_dim*2, fusion_dim)
        # self.multi_modal = nn.Linear(fusion_dim+ct_dim, fusion_dim)
        self.multi_modal_classifiers = nn.Linear(fusion_dim, n_classes)

    def forward(self, h, ct):
        # print(h.shape)
        h = h.squeeze()
        h = h.float()
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, h)
        logits = self.classifiers(M)

        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d):

                ct = ct.squeeze(0)
                ct = layer(ct)
                ct = ct.unsqueeze(0)
            else:
                ct = layer(ct)
        fusion = torch.cat([logits, ct], dim=1)
        fusion = self.multi_modal(fusion)
        fusion_logits = self.multi_modal_classifiers(fusion)


        return fusion_logits
    

class CLAM_SB_multimodal2(nn.Module):
    def __init__(self, size_arg = "small", dropout = False, n_classes=2, ct_dim=128, fusion_dim=128):
        super(CLAM_SB_multimodal2, self).__init__()
        self.size_dict = {"small": [512, 256, 128], "big": [256, 192, 128]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(dropout))
        attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], fusion_dim)
        self.n_classes = n_classes

        # initialize_weights(self)
        # ct modalities
        self.preprocess_ct = nn.Linear(ct_dim, fusion_dim)

        self.multi_modal = nn.Linear(fusion_dim*2, fusion_dim)
        # self.multi_modal = nn.Linear(fusion_dim+ct_dim, fusion_dim)
        self.multi_modal_classifiers = nn.Linear(fusion_dim, n_classes)

    def forward(self, h, ct):
        # print(h.shape)
        h = h.squeeze()
        h = h.float()
        A, h = self.attention_net(h)  # NxK
        # print(A.shape, h.shape)
        A = torch.transpose(A, 1, 0)  # KxN

        A = F.softmax(A, dim=1)  # softmax over N

        # print(A.shape, h.shape)

        M = torch.mm(A, h)
        logits = self.classifiers(M)

        fusion = torch.cat([logits, ct], dim=1)
        fusion = self.multi_modal(fusion)
        fusion_logits = self.multi_modal_classifiers(fusion)


        return fusion_logits
    

class CLAM_SB_multimodal2_patho(nn.Module):
    def __init__(self, size_arg = "small", dropout = False, n_classes=2, ct_dim=128, fusion_dim=128):
        super(CLAM_SB_multimodal2_patho, self).__init__()
        self.size_dict = {"small": [512, 256, 128], "big": [256, 192, 128]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(dropout))
        attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        self.n_classes = n_classes


    def forward(self, h, ct):
        # print(h.shape)
        h = h.squeeze()
        h = h.float()
        A, h = self.attention_net(h)  # NxK
        # print(A.shape, h.shape)
        A = torch.transpose(A, 1, 0)  # KxN

        A = F.softmax(A, dim=1)  # softmax over N

        # print(A.shape, h.shape)

        M = torch.mm(A, h)
        logits = self.classifiers(M)
        return logits





class CLAM_SB_multimodal2_ct(nn.Module):
    def __init__(self, n_classes=2, input_dim=2164*2, dim=128, activation='Identity', depth=3, dropout=0.):
        super(CLAM_SB_multimodal2_ct, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_dim, dim))
        for _ in range(depth - 1):
            self.layers.append(nn.Linear(dim, dim))
            #self.layers.append(nn.BatchNorm1d(dim))
            self.layers.append(eval("nn." + activation)())
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(dim, n_classes))

    def forward(self, h, ct):
        x = ct
        for layer in self.layers:
            x = layer(x)
        return x
    

class CLAM_SB_multimodal2_interprete(nn.Module):
    def __init__(self, size_arg = "small", dropout = False, n_classes=2, ct_dim=128, fusion_dim=128):
        super(CLAM_SB_multimodal2_interprete, self).__init__()
        self.size_dict = {"small": [512, 256, 128], "big": [256, 192, 128]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(dropout))
        attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], fusion_dim)
        self.n_classes = n_classes

        self.preprocess_ct = nn.Linear(ct_dim, fusion_dim)

        self.multi_modal = nn.Linear(fusion_dim*2, fusion_dim)
        # self.multi_modal = nn.Linear(fusion_dim+ct_dim, fusion_dim)
        self.multi_modal_classifiers = nn.Linear(fusion_dim, n_classes)

    def forward(self, h, ct, return_interprete=True):
        h = h.squeeze()
        h = h.float()
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, h)
        logits = self.classifiers(M)

        fusion = torch.cat([logits, ct], dim=1)
        fusion = self.multi_modal(fusion)
        fusion_logits = self.multi_modal_classifiers(fusion)

        if return_interprete:
            return fusion_logits, A
        return fusion_logits


class CLAM_SB_multimodal_mul(nn.Module):
    def __init__(self, size_arg = "small", dropout = False, n_classes=2, ct_dim=128, fusion_dim=128):
        super(CLAM_SB_multimodal_mul, self).__init__()
        self.size_dict = {"small": [512, 256, 128], "big": [256, 192, 128]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(dropout))
        attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], fusion_dim)
        self.n_classes = n_classes
        prect = [nn.Linear(ct_dim, fusion_dim)]
        self.preprocess_ct = nn.Sequential(*prect)
        self.multi_modal_classifiers = nn.Linear(fusion_dim, n_classes)

        # initialize_weights(self)

    def forward(self, h, ct):
        # print(h.shape)
        h = h.squeeze()
        h = h.float()
        A, h = self.attention_net(h)  # NxK
        # print(A.shape, h.shape)
        A = torch.transpose(A, 1, 0)  # KxN

        A = F.softmax(A, dim=1)  # softmax over N

        # print(A.shape, h.shape)

        M = torch.mm(A, h)

        #fusion = torch.mul(M, ct)
        ct = self.preprocess_ct(ct)

        logits = self.classifiers(M)
        fusion = torch.mul(logits, ct)
        fusion_logits = self.multi_modal_classifiers(fusion)

        return fusion_logits
    
# dual-path FC
class dual_path_FC(nn.Module):
    def __init__(self, input_dim=128, output_dim=128):
        super(dual_path_FC, self).__init__()
        self.FC1 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
            nn.Linear(input_dim, output_dim)
        )
        self.FC2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x1 = self.FC1(x)
        x2 = self.FC2(x)
        return x1+x2

class CLAM_SB_multimodal_mul2(nn.Module):
    # def __init__(self, size_arg = "small", dropout = False, n_classes=2, ct_dim=128, fusion_dim=128):
    def __init__(self, size_arg = "small", dropout = False, n_classes=2, ct_dim=192, fusion_dim=128):
        super(CLAM_SB_multimodal_mul2, self).__init__()
        self.size_dict = {"small": [512, 256, 128], "big": [256, 192, 128]}
        # self.size_dict = {"small": [512, 384, 128], "big": [256, 192, 128]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        # fc = [nn.Linear(size[0], size[1]), nn.LeakyReLU()]
        # fc = [nn.Linear(size[0], size[1]), nn.SiLU()]
        if dropout:
            fc.append(nn.Dropout(dropout))
        attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], fusion_dim)
        self.n_classes = n_classes
        prect = [nn.Linear(ct_dim, ct_dim), nn.Sigmoid(), nn.Linear(ct_dim, 128)]

        self.preprocess_ct = nn.Sequential(*prect)
        self.multi_modal_classifiers = nn.Sequential(
            nn.Linear(fusion_dim+128, 128),
            nn.Sigmoid(),
            nn.Linear(128, n_classes)
        )

        initialize_weights(self)

    def forward(self, h, ct):
        # print(h.shape)
        h = h.squeeze()
        h = h.float()
        A, h = self.attention_net(h)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, h)

        ct = self.preprocess_ct(ct)
        logits = self.classifiers(M)
        fusion = torch.cat([logits, ct], dim=1)
        fusion_logits = self.multi_modal_classifiers(fusion)

        return fusion_logits


class CLAM_SB_multimodal_mul2_interprete(nn.Module):
    def __init__(self, size_arg = "small", dropout = False, n_classes=2, ct_dim=192, fusion_dim=128):
        super(CLAM_SB_multimodal_mul2_interprete, self).__init__()
        self.size_dict = {"small": [512, 256, 128], "big": [256, 192, 128]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        
        if dropout:
            fc.append(nn.Dropout(dropout))
        attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_classes = 1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], fusion_dim)
        self.n_classes = n_classes
        
        prect = [nn.Linear(ct_dim, ct_dim), nn.Sigmoid(), nn.Linear(ct_dim, 128)]
        self.preprocess_ct = nn.Sequential(*prect)
        self.multi_modal_classifiers = nn.Sequential(
            nn.Linear(fusion_dim+128, 128),
            nn.Sigmoid(),
            nn.Linear(128, n_classes)
        )

        initialize_weights(self)

    def forward(self, h, ct, return_interprete=True):
        # print(h.shape)
        h = h.squeeze()
        h = h.float()
        A, h = self.attention_net(h)  # NxK
        # print(A.shape, h.shape)
        A = torch.transpose(A, 1, 0)  # KxN

        A = F.softmax(A, dim=1)  # softmax over N

        # print(A.shape, h.shape)

        M = torch.mm(A, h)

        ct = self.preprocess_ct(ct)
        '''for layer in self.layers:
            ct = layer(ct)'''
        logits = self.classifiers(M)
        #fusion = torch.mul(logits, ct)
        fusion = torch.cat([logits, ct], dim=1)
        
        fusion_logits = self.multi_modal_classifiers(fusion)

        if return_interprete:
            return fusion_logits, A
        return fusion_logits
    
class Transformer_patho(nn.Module):
    def __init__(self, n_classes=2, dim=256, depth=2, heads=4, dim_head=64, mlp_dim=256, dropout=0.1):
        super(Transformer_patho, self).__init__()
        self.pre_transform = nn.Linear(512, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token_g = nn.Parameter(torch.randn(1, 1, dim))
        self.basetransformer = TransformerBlocks(dim, depth, heads, dim_head, mlp_dim, dropout=dropout)
        self.grouptransformer = TransformerBlocks(dim, depth, heads, dim_head, mlp_dim, dropout=dropout)
        self.classifier = nn.Linear(dim, n_classes)

    def forward(self, x, ct):
        x = self.pre_transform(x)
        # 1 * n_tokens * dim -> n_tokens/k * k * dim
        k = 4
        # padding
        n = x.size(1)
        pad = k - n % k
        x = F.pad(x, (0, 0, 0, pad))
        x = x.view(-1, k, x.size(-1)).squeeze(0)
        x = torch.cat((self.cls_token.expand(x.size(0), -1, -1), x), dim=1)
        x = self.basetransformer(x)
        x = x[:, 0]
        # group transformer
        # cat with group token
        x = x.unsqueeze(0)
        x = torch.cat((self.cls_token_g.expand(x.size(0), -1, -1), x), dim=1)
        x = self.grouptransformer(x)
        x = x[:, 0]
        x = self.classifier(x)
        return x
    
class DiffTransformer_patho(nn.Module):
    def __init__(self, n_classes=2, dim=256, depth=2, heads=4, dim_head=64, mlp_dim=256, dropout=0.1):
        super(DiffTransformer_patho, self).__init__()
        self.pre_transform = nn.Linear(512, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = DiffTransformerBlocks(dim, depth, heads, dim_head, mlp_dim, dropout=dropout)
        self.classifier = nn.Linear(dim, n_classes)

    def forward(self, x, ct):
        x = self.pre_transform(x)
        x = torch.cat((self.cls_token.expand(x.size(0), -1, -1), x), dim=1)
        x = self.transformer(x)
        x = x[:, 0]
        x = self.classifier(x)
        return x


def initialize_weights(model, init_mode='xavier', activation='relu'):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init_mode == 'xavier':
                nn.init.xavier_normal_(m.weight)
            elif init_mode == 'kaiming':
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)