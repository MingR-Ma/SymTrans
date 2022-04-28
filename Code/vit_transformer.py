import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath
from functools import reduce


def pair(t):
    """
    Get each dimension numbers.
    :param t: tuple
        A image shape '(depth, height, width)', this is specific for 3D images.
    :return: int, int ,int
        Three elements represent 'depth','height','width'.
    """
    return t if isinstance(t, tuple) else (t, t, t)


class SepConv3d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, ):
        """
        Depthwise separable convolution.
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        """
        super(SepConv3d, self).__init__()
        self.depthwise = torch.nn.Conv3d(
            in_channels, in_channels, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=in_channels)
        self.pointwise = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, in_chans, padding, embed_dim, stirde=2, bias=False):
        """
        Split image into patches and then embed them.
        parameters
        ----------
        :param img_size: Original image size
        :param patch_size: Split image into patches
        :param in_chans: Original image channels
        :param embed_dim: the number you want the vector embed you want.

        Attributes
        ----------
        n_patches: int
            Number of patches inside of our image.

        proj: nn.Conv2d
            Convolutional layer that does both the splitting into patches
            and their embedding.

        """
        super(PatchEmbed, self).__init__()

        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_chans,
            self.embed_dim,
            kernel_size=patch_size,
            stride=stirde,
            padding=padding,
            bias=bias
        )
        self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

    def forward(self, x):
        """
        Run forward pass.
        :param x: torch.Tensors
                Shape '(n_samples, in_chans, img_size, img_size)'.
        :return: torch.Tensor
                Shape '(n_samples, n_patches, embed_dim)'.
        """
        x = self.proj(x)  # (b:n_samples, c:embed_dim, h:n_patches**0.5, w:n_patches**0.5)
        x = rearrange(x, 'b c d h w -> b (d h w) c')  # (b:n_samples, l:n_patches, c:embed_dim)
        x = self.norm(x)

        return x

# The efficient proposed multi-head self-attention
class DSAttention(nn.Module):
    def __init__(
            self, dim, image_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
            sr_ratio=1, kernel_size=3, q_stride=1
    ):
        super().__init__()
        self.num_heads = num_heads
        self.img_size = image_size
        head_dim = dim // num_heads
        pad = (kernel_size - q_stride) // 2
        inner_dim = dim
        self.scale = head_dim ** -0.5  # not used

        self.q = SepConv3d(dim, inner_dim, kernel_size, q_stride, pad)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio + 1, stride=sr_ratio, padding=sr_ratio // 2, groups=dim,
                                bias=False)
            self.sr_norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        B, N, C = x.shape
        b, n, _, num_heads = *x.shape, self.num_heads
        xq = rearrange(
            x, 'b (l w d) n -> b n l w d',
            l=self.img_size[0], w=self.img_size[1], d=self.img_size[2]
        )
        q = self.q(xq)
        q = rearrange(q, 'b (h d) l w k -> b h (l w k) d', h=num_heads)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, self.img_size[0], self.img_size[1], self.img_size[2])
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.sr_norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# The standard multi-head self-attention
class Attention(nn.Module):
    def __init__(self, dim, n_heads, qkv_bias=False, attn_p=0., proj_p=0.):
        """
        Attention mechanism
        :param dim: int
            The input and out dimension of per token features.
            For now, we regard tokens as the patches.
        :param n_heads: int
            number of attention heads.
        :param qkv_bias: bool
            If True then we include bias to the query, key and value projections.
        :param attn_p: float
            Dropout probability applied to the query, key and value tensors.
        :param proj_p: float
            Dropout probability applied to the output tensors.

        Attributes
        ----------
        scale: float
            Normalizing constant for the dot product.
        qkv: nn.Linear
            Linear projection for the query, key and value.
        proj: nn.Linear
            Linear mapping that takes in the concatenated out put of all attention
            heads and maps it into a new space.
        attn_drop, proj_drop: nn.Dropout
            Dropout layers.
        """
        super(Attention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(in_features=dim, out_features=dim * 3, bias=qkv_bias)
        # self.merge = nn.Conv1d(in_channels=dim * 2, out_channels=dim * 2, bias=False)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(in_features=dim, out_features=dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """
        Run forward pass.
        :param x: torch.Tensor
            Shape '(n_samples, n_patches+1, dim)'.
        :return: torch.Tensor
            Shape '(n_samples, n_patches+1, dim)'.
        """
        n_samples, n_tokens, dim = x.shape  # (n_samples,n_patched+1,embed_dim)

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches+1,3*dim) get query,key,value.
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_samples,n_patches+1,qkv,n_heads,head_dim)

        qkv = rearrange(
            qkv,
            'n_samples n_tokens qkv n_heads head_dim -> qkv n_samples n_heads n_tokens head_dim'
        )  # (n_samples,n_heads,n_patches+1,head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples,n_heads,head_dim,n_patches+1),
        # this is in order to compute the dot pruduct

        dp = (q @ k_t) * self.scale  # (n_samples,n_heads,n_patches+1,n_patches+1)

        attn = dp.softmax(dim=-1)  # (n_samples,n_heads,n_patches+1,n_patches+1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples,n_heads,n_patches+1,head_dim)
        weighted_avg = weighted_avg.transpose(1, 2)
        # (n_samples,n_patches+1,n_heads,head_dim)

        weighted_avg = rearrange(
            weighted_avg, 'n_samples n_patches n_heads head_dim -> n_samples n_patches (n_heads head_dim)'
        )  # (n_samples,n_patches+1,dim)

        x = self.proj(weighted_avg)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p):
        """
        Multilayer perceptron.
        :param in_features: int
            Number of input features.
        :param hidden_features: int
            Number of output features.
        :param out_features: int
            Number of output features.
        :param p:
            Dropout probability.

        Attribute
        ---------
        fc: nn.Linear
            The first linear layer.
        act: nn.GELU
            GELU activation function.
        fc2: nn.Linear
            The second linear layer.
        drop: nn.Dropout
            Dropout layer.
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(in_features=hidden_features, out_features=out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """
        Run forward pass.
        :param x: torch.Tensor
            Shape '(n_samples, n_patches+1, in_features)'.
        :return: x: torch.Tensor
            Shape '(n_samples, n_patches+1, out_features)'
        """
        x = self.fc1(x)  # (n_samples, n_patches+1, hidden_features)
        x = self.act(x)  # (n_samples, n_patches+1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches+1, hidden_features)

        x = self.fc2(x)  # (n_samples, n_patches+1, out_features)
        x = self.drop(x)  # (n_samples, n_patches+1, out_features)

        return x


class Block(nn.Module):
    def __init__(self,
                 dim, image_size, n_heads, sr_ratio, drop_path_ratio=0., mlp_ratio=4.0,
                 qkv_bias=True, p=0., attn_p=0., proj_drop=0
                 ):
        """
        Transformer block
        :param dim: int
            Embedding dimension

        :param n_heads: int
            number of attention heads.

        :param mlp_ratio: float
            Determines the hidden dimension size of the 'MLP' module with respect
            to 'dim'.

        :param qkv_bias: bool
            It Ture then we include bias to the query,key and value projections.

        :param p, attn_p: float
            Dropout probability.

        Attributes:
        ----------
        norm1,norm2: layerNorm
            layer normalization.

        attn: Attention
            Attention module.

        mlp: MLP
            MLP module.
        """
        super(Block, self).__init__()

        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)

        self.attn = DSAttention(
            dim, image_size, num_heads=n_heads, qkv_bias=qkv_bias,
            sr_ratio=sr_ratio, proj_drop=proj_drop, attn_drop=attn_p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=hidden_features,
            out_features=dim,
            p=p
        )

    def forward(self, x):
        """
        Run forward pass.
        :param x: torch.Tensor
            Shape '(n_samples, n_patches+1, dim)'
        :return x: torch.Tensor
            shape '(n_samples, n_patches+1,dim)'
        """

        x = x + self.drop_path(
            self.attn(self.norm1(x))
        )
        x = x + self.drop_path(
            self.mlp(self.norm2(x))
        )

        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 img_size, patch_size, embed_dim, sr_ratio,
                 depth, n_heads, drop_path_ratio=0., mlp_ratio=4., qkv_bias=True, p=0., att_p=0.
                 ):
        """
        Simplified implementation of the Vision transformer.
        :param img_size: int
            Both height and the width of the image (it is a square or not).

        :param patch_size: int -> tuple
            in original, it's should be square. Here, we modify it to apply
             non-square patch size.
            ---------------------------------------------------------------------
            Both height and the width of the patch (it is a square or not).

        :param in_chans: int
            Number of input channels (e.g. RGB in_chans=3, MRI scans in_chans=1)

        :param n_class: int
            Number of classes.

        :param embed_dim: int
            Dimensionality of the token/patch embeddings.

        :param depth: int
            Number of blocks.(Iterations of transformer block indicate how many
            transformer block we use).

        :param n_heads: int
            Number of attention heads.

        :param mlp_ratio: float
            Determines the hidden dimension of the 'MLP' module ( Papers shows
            that the in_features times 4.0 (i.e mlp_ratio=4.0 is better)).

        :param qkv_bias: bool
            If True then we include bias to the query, key and value.

        :param p: float
            Dropout probability.
        :param att_p:
            Dropout probability.

        Attributions:
        ------------trunc_normal_
        patch_embed: PatchEmbed
            Instance of 'PatchEmbed' layer.

        cls_token: nn.Parameter
            Learnable parameter that will represent the first token in the sequence.
            It has 'embed_dim' elements.

        pos_emb: nn.Parameter
            Positional embedding of the cls token + all the patches.
            It has '(n_patches+1) * embed_dim' elements.

        pos_drop: nn.Dropout
            Dropout layer.

        blocks: nn.ModuleList
            List of 'Block' modules.

        norm: nn.LayerNorm
            Layer normalization.
        """
        super(VisionTransformer, self).__init__()

        self.img_size = img_size
        # print(self.img_size)
        self.patch_size = patch_size
        img_depth, img_height, img_width = pair(img_size)
        patch_depth, patch_height, patch_width = pair(patch_size)
        self.embed_dim = embed_dim
        self.n_patches = img_height * img_width * img_depth

        self.pos_drop = nn.Dropout(p=p)

        # second: create the transformer
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    image_size=self.img_size,
                    n_heads=n_heads,
                    sr_ratio=sr_ratio,
                    drop_path_ratio=drop_path_ratio,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=att_p,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    # for class, this followed is not used
    # self.head = nn.Linear(embed_dim, n_class)

    def forward(self, x):
        """
        Run the forward pass.
        :param x: torch.Tensor
            Shape '(n_samples, in_chans ,img_size, img_size)'.
        :return: x: torch.Tensor (For classification)
            Logits over all the classes - '(n_samples, n_classes)'
                 x: torch.Tensor (For registration)
            Shape '(n_samples, n_deformation_axes, img_depth, img_height, img_width)'.
        """

        n_samples = x.shape[0]

        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x


class PatchExpanding(nn.Module):
    def __init__(self, dim, input_shape, scale_factor=8, bias=False):
        """
        Expand operation in decoder.
        :param dim: input token channels for expanding.
        :param scale_factor: the expanding scale for token channels.
        """
        super(PatchExpanding, self).__init__()
        self.dim = dim
        self.input_shape = input_shape

        self.expander = nn.Sequential(
            nn.Linear(self.dim, scale_factor * self.dim, bias=bias),
            Rearrange('b (D H W) c -> b D H W c', D=self.input_shape[0], H=self.input_shape[1],W=self.input_shape[2]),
            Rearrange('b D H W (h d w c) -> b (D d) (H h) (W w) c', d=2,h=2, w=2, c=self.dim),
            Rearrange('b D H W c -> b (D H W) c'),
            nn.Linear(self.dim, self.dim // 2)
        )
        self.norm = nn.LayerNorm(self.dim // 2, eps=1e-6)

    def forward(self, x):
        """
        Run forward pass.
        :param x: torch.Tensor
            Input tokens.
        :return: x: torch.Tensor
            Output expanded tokens.
        """
        x = self.norm(self.expander(x))

        return x


class VectorDownSample(nn.Module):
    def __init__(self, shape, dim, image_dim=2):
        """

        :param size: tuple, image size.
        :param dim: Int, vector dimension.
        :param image_dim: Int, image dimension, default: 2 for 2D image, 3 for 3D image.
        """

        self.dim = dim
        self.size = reduce(lambda x, y: x * y, shape)

        super(VectorDownSample, self).__init__()
        self.downsample = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2),
            Rearrange('B L C -> B C L'),
            nn.Linear(self.size, self.size // (2 ** image_dim)),
            Rearrange('B C L -> B L C'),
            nn.LayerNorm(self.dim * 2, eps=1e-6)
        )

    def forward(self, x):
        """

        :param x: shaped in [B,L,C]
        :return: downsampled_vector, shaped in [B,L,C]
        """
        downsampled_vector = self.downsample(x)
        return downsampled_vector


class VectorReduce(nn.Module):
    def __init__(self, dim):
        super(VectorReduce, self).__init__()
        self.dim = dim
        self.reduce = nn.Linear(self.dim, self.dim // 2)
        self.norm = nn.LayerNorm(self.dim // 2, eps=1e-6)

    def forward(self, x):
        x = self.reduce(x)
        x = self.norm(x)
        return x


class SkipConnection(nn.Module):
    def __init__(self, dim, bias=False):
        """

        :param dim: each concatenating tokens dim.
        """
        super(SkipConnection, self).__init__()
        self.fusion = nn.Linear(2 * dim, dim, bias=bias)

    def forward(self, x, y):
        x = torch.cat([x, y], -1)
        x = self.fusion(x)

        return x


class SkipConnection_v2(nn.Module):
    def __init__(self, dim, input_shape, bias=False):
        """

        :param dim: each concatenating tokens dim.
        """
        super(SkipConnection_v2, self).__init__()
        self.input_shape = input_shape
        self.fusion = nn.Conv3d(2 * dim, dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x, y):
        x = rearrange(x, 'b (D H W) c -> b c D H W', D=self.input_shape[0], H=self.input_shape[1],
                      W=self.input_shape[2])
        y = rearrange(y, 'b (D H W) c -> b c D H W', D=self.input_shape[0], H=self.input_shape[1],
                      W=self.input_shape[2])

        x = torch.cat([x, y], 1)
        x = self.fusion(x)

        return x
