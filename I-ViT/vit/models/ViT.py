import torch
import torch.nn as nn
import torch.nn.functional as F
from .Transformer import TransformerModel
from .PositionalEncoding import (
    FixedPositionalEncoding,
    LearnedPositionalEncoding,
    LearnedPositionalEncoding2,
    LearnedPositionalEncoding3,
)

from torchvision.models.resnet import BasicBlock

__all__ = ['VitsCNN_pos', 'Vit']


    
class VitsCNN_pos(nn.Module):
    def __init__(
        self,
        out_dim=2,
        num_nuclei = 50,
        embedding_dim = 84,
        num_heads =28,
        num_layers=2,
        hidden_dim=128,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,
        nuclues_size = 32,
        use_representation=False,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    ):
        super(VitsCNN_pos, self).__init__()

        #assert embedding_dim % num_heads == 0
        #assert img_dim % patch_dim == 0
        self.num_nuclei = num_nuclei
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        #self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation
        self.seq_length = 5
        self.nuclues_size = nuclues_size
        
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding2(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )
        self.position_encoding_xy = LearnedPositionalEncoding3(
                201, self.embedding_dim, 201
            )
        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)
        if use_representation:
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.mlp_head = nn.Linear(embedding_dim, out_dim)

        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1)
        
        if self.nuclues_size == 16:
            self.fc1 = nn.Linear(16 * 2 * 2, 120)
        elif self.nuclues_size == 32:
            self.fc1 = nn.Linear(16 * 6 * 6, 120)
        elif self.nuclues_size == 64:
            self.fc1 = nn.Linear(16 * 14 * 14, 120)
        self.fc2 = nn.Linear(120, 84)
        
        self.to_cls_token = nn.Identity()

    def forward(self, x, cls,pos):
        # grade embedding
        px = pos[:,:,0]
        py = pos[:,:,1]
        t = cls[:,:self.num_nuclei]
        x=x.to(torch.float32)

        batch, num, c, w, h = x.shape
        x = x.reshape(batch*num, c, w, h)
        # A small cnn for nuclei feature extraction
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.reshape(batch, num, self.embedding_dim)
        # x is a n * num_nuclei * channel
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        #核类别嵌入
        x = self.position_encoding(x,t)
        #位置嵌入
        x = self.position_encoding_xy(x,px,py)
        
        x = self.pe_dropout(x)

        x = x.to(torch.float32)
        x = self.transformer(x)
        x = self.pre_head_ln(x)
        x = self.to_cls_token(x[:, 0])
        x = self.mlp_head(x)
        x = F.log_softmax(x, dim=-1)

        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    
    
    
    
    



    

class Vit(nn.Module):
    def __init__(
        self,
        out_dim=2,
        num_nuclei = 200,
        embedding_dim = 10,
        num_heads =2,
        num_layers=2,
        hidden_dim=128,
        dropout_rate=0.1,
        attn_dropout_rate=0.0,
        use_representation=False,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    ):
        super(Vit, self).__init__()

        #assert embedding_dim % num_heads == 0
        #assert img_dim % patch_dim == 0
        self.num_nuclei = num_nuclei
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        #self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation
        self.seq_length = 5
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.flatten_dim = 10
        self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding2(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)
        if use_representation:
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.mlp_head = nn.Linear(embedding_dim, out_dim)

        self.to_cls_token = nn.Identity()

    def forward(self, x):
        n, h, w = x.shape
        t = x[:,:,10]
        x = x[:,:,0:10]
      
        x=x.to(torch.float32)
#         x = self.linear_encoding(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.position_encoding(x,t)
        x = self.pe_dropout(x)
        x=x.to(torch.float32)
        x = self.transformer(x)
        x = self.pre_head_ln(x)
        x = self.to_cls_token(x[:, 0])
        x = self.mlp_head(x)
        x = F.log_softmax(x, dim=-1)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        out_dim,
        num_channels,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        use_representation=True,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    ):
        super(VisionTransformer, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation

        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches + 1
        self.flatten_dim = patch_dim * patch_dim * num_channels
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)
        if use_representation:
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.mlp_head = nn.Linear(embedding_dim, out_dim)

        if self.conv_patch_representation:
            self.conv_x = nn.Conv2d(
                self.num_channels,
                self.embedding_dim,
                kernel_size=(self.patch_dim, self.patch_dim),
                stride=(self.patch_dim, self.patch_dim),
                padding=self._get_padding(
                    'VALID', (self.patch_dim, self.patch_dim),
                ),
            )
        else:
            self.conv_x = None

        self.to_cls_token = nn.Identity()

    def forward(self, x):
        n, c, h, w = x.shape
        print(x.shape)
        if self.conv_patch_representation:
            # combine embedding w/ conv patch distribution
            x = self.conv_x(x)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.flatten_dim)
        else:
            x = (
                x.unfold(2, self.patch_dim, self.patch_dim)
                .unfold(3, self.patch_dim, self.patch_dim)
                .contiguous()
            )
            print(x.shape)
            x = x.view(n, c, -1, self.patch_dim ** 2)
            print(x.shape)
            x = x.permute(0, 2, 3, 1).contiguous()
            print(x.shape)
            x = x.view(x.size(0), -1, self.flatten_dim)
            print(x.shape)
            x = self.linear_encoding(x)
            print(x.shape)
        
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        print(cls_tokens.shape)
        x = torch.cat((cls_tokens, x), dim=1)
        print(x.shape)
        x = self.position_encoding(x)
        print(x.shape)
        x = self.pe_dropout(x)
        print(x.shape)
        print('transformer')
        # apply transformer
        x = self.transformer(x)
        x = self.pre_head_ln(x)
        x = self.to_cls_token(x[:, 0])
        print(x.shape)
        x = self.mlp_head(x)
        x = F.log_softmax(x, dim=-1)
        print(x.shape)
        print('next')
        return x

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)


