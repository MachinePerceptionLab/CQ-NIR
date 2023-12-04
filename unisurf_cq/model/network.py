import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    ''' Network class containing occupanvy and appearance field
    
    Args:
        cfg (dict): network configs
    '''

    def __init__(self, cfg, **kwargs):
        super().__init__()
        out_dim = 4
        dim = 3
        self.num_layers = cfg['num_layers']
        hidden_size = cfg['hidden_dim']
        self.octaves_pe = cfg['octaves_pe']
        self.octaves_pe_views = cfg['octaves_pe_views']
        self.skips = cfg['skips']
        self.rescale = cfg['rescale']
        self.feat_size = cfg['feat_size']
        geometric_init = cfg['geometric_init'] 

        bias = 0.6

        # init pe
        dim_embed = dim*self.octaves_pe*2 + dim
        dim_embed_view = dim + dim*self.octaves_pe_views*2 + dim + dim + self.feat_size 
        self.transform_points = PositionalEncoding(L=self.octaves_pe)
        self.transform_points_view = PositionalEncoding(L=self.octaves_pe_views)

        ### geo network
        dims_geo = [dim_embed]+ [ hidden_size if i in self.skips else hidden_size for i in range(0, self.num_layers)] + [self.feat_size+1] 
        self.num_layers = len(dims_geo)
        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skips:
                out_dim = dims_geo[l + 1] - dims_geo[0]
            else:
                out_dim = dims_geo[l + 1]

            lin = nn.Linear(dims_geo[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims_geo[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif self.octaves_pe > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif self.octaves_pe > 0 and l in self.skips:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims_geo[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            
            lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

        ## appearance network
        dims_view = [dim_embed_view]+ [ hidden_size for i in range(0, 4)] + [3]

        self.num_layers_app = len(dims_view)

        for l in range(0, self.num_layers_app - 1):
            out_dim = dims_view[l + 1]
            lina = nn.Linear(dims_view[l], out_dim)
            lina = nn.utils.weight_norm(lina)
            setattr(self, "lina" + str(l), lina)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def infer_occ(self, p):
        pe = self.transform_points(p/self.rescale)
        x = pe
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.skips:
                x = torch.cat([x, pe], -1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)     
        return x
    
    def infer_app(self, points, normals, view_dirs, feature_vectors):
        rendering_input = torch.cat([points, view_dirs, normals.squeeze(-2), feature_vectors], dim=-1)
        x = rendering_input
        for l in range(0, self.num_layers_app - 1):
            lina = getattr(self, "lina" + str(l))
            x = lina(x)
            if l < self.num_layers_app - 2:
                x = self.relu(x)
        x = self.tanh(x) * 0.5 + 0.5
        return x

    def pts_loc(self, pts):
        bs, npoints, _ = pts.shape
        xmin = -4.0
        xmax = 4.0
        res = 51200
        cube_size = 1 / (res - 1)

        # normalize coords for interpolation
        pts = (pts - xmin) / (xmax - xmin)  # normalize to 0 ~ 1
        # pts = pts.clamp(min=1e-6, max=1 - 1e-6)
        ind0 = (pts / cube_size).floor()  # grid index (bs, npoints, 3)

        # get 8 neighbors
        offset = torch.Tensor([0])
        grid_x, grid_y, grid_z = torch.meshgrid(*tuple([offset] * 3))
        neighbor_offsets = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        neighbor_offsets = neighbor_offsets.reshape(-1, 3)  # 1*3
        nneighbors = neighbor_offsets.shape[0]
        neighbor_offsets = neighbor_offsets.type(torch.cuda.FloatTensor)  # shape==(1, 3)

        # get neighbor 8 latent codes
        neighbor_indices = ind0.unsqueeze(2) + neighbor_offsets[None, None, :, :]  # (bs, npoints, 1, 3)
        neighbor_indices = neighbor_indices.type(torch.cuda.LongTensor)
        neighbor_indices = neighbor_indices.reshape(bs, -1, 3)  # (bs, npoints*1, 3)

        # get the tri-linear interpolation weights for each point
        xyz0 = ind0 * cube_size  # (bs, npoints, 3)
        xyz0_expand = xyz0.unsqueeze(2).expand(bs, npoints, nneighbors, 3)  # (bs, npoints, nneighbors, 3)
        xyz_neighbors = xyz0_expand + neighbor_offsets[None, None, :, :] * cube_size + cube_size/2.0

        neighbor_offsets_oppo = 1 - neighbor_offsets
        xyz_neighbors_oppo = xyz0.unsqueeze(2) + neighbor_offsets_oppo[None,
                                                                       None, :, :] * cube_size  # bs, npoints, 1, 3
        dxyz = (pts.unsqueeze(2) - xyz_neighbors_oppo).abs() / cube_size
        weight = dxyz[:, :, :, 0] * dxyz[:, :, :, 1] * dxyz[:, :, :, 2]          # bs, npoints, 1

        xyz_neighbors = xyz_neighbors*2*xmax - xmax
        
        return weight, xyz_neighbors

    def gradient(self, p):
        with torch.enable_grad():
            if len(p.shape) !=2:
                p = p.squeeze()
            _, center= self.pts_loc(p[None])
            p = center.squeeze()
            p.requires_grad_(True)
            y = self.infer_occ(p)[...,:1]
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=p,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True, allow_unused=True)[0]
            return gradients.unsqueeze(1)

    def forward(self, p, ray_d=None, only_occupancy=False, return_logits=False,return_addocc=False, noise=False, **kwargs):
        if len(p.shape) !=2:
            p = p.squeeze()
        _, center= self.pts_loc(p[None])
        x = self.infer_occ(center.squeeze())

        if only_occupancy:
            return self.sigmoid(x[...,:1] * -10.0)
        elif ray_d is not None:
            
            input_views = ray_d / torch.norm(ray_d, dim=-1, keepdim=True)
            input_views = self.transform_points_view(input_views)
            normals =  self.gradient(p)
            #normals = n / (torch.norm(n, dim=-1, keepdim=True)+1e-6)
            rgb = self.infer_app(center.squeeze(), normals, input_views, x[...,1:])
            if return_addocc:
                if noise:
                    return rgb, self.sigmoid(x[...,:1] * -10.0 )
                else: 
                    return rgb, self.sigmoid(x[...,:1] * -10.0 )
            else:
                return rgb
        elif return_logits:
            return -1*x[...,:1]


class PositionalEncoding(object):
    def __init__(self, L=10):
        self.L = L
    def __call__(self, p):
        pi = 1.0
        p_transformed = torch.cat([torch.cat(
            [torch.sin((2 ** i) * pi * p), 
             torch.cos((2 ** i) * pi * p)],
             dim=-1) for i in range(self.L)], dim=-1)
        return torch.cat([p, p_transformed], dim=-1)
