import torch
import torch.nn as nn

import numpy

from torch_geometric.utils import scatter

class Envelope(nn.Module):
    def __init__(self, exponent):
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return (1.0 / x + a * x_pow_p0 + b * x_pow_p1 +
                c * x_pow_p2) * (x < 1.0).to(x.dtype)

class BesselBasisLayer(torch.nn.Module):
    def __init__(self, num_radial, cutoff = 5.0,
                 envelope_exponent5):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = torch.nn.Parameter(torch.empty(num_radial))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)
        self.freq.requires_grad_()

    def forward(self, dist):
        dist = dist.unsqueeze(-1) / self.cutoff
        return self.envelope(dist) * (self.freq * dist).sin()

class SphericalBasisLayer(torch.nn.Module):
    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        envelope_exponent: int = 5,
    ):
        super().__init__()
        import sympy as sym

        from torch_geometric.nn.models.dimenet_utils import (
            bessel_basis,
            real_sph_harm,
        )

        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist: Tensor, angle: Tensor, idx_kj):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out

class EmbeddingBlock(torch.nn.Module):
    def __init__(self, num_radial: int, hidden_channels: int, act:
        super().__init__()
        self.act = act

        self.emb = nn.Embedding(95, hidden_channels)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels)
        self.lin = nn.Linear(3 * hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        self.lin_rbf.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x: Tensor, rbf: Tensor, i: Tensor, j: Tensor) -> Tensor:
        x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act):
        super().__init__()
        self.act = act
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class InteractionPPBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        int_emb_size: int,
        basis_emb_size: int,
        num_spherical: int,
        num_radial: int,
        num_before_skip: int,
        num_after_skip: int,
        act: Callable,
    ):
        super().__init__()
        self.act = act

        # Transformation of Bessel and spherical basis representations:
        self.lin_rbf1 = Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = Linear(basis_emb_size, hidden_channels, bias=False)

        self.lin_sbf1 = Linear(num_spherical * num_radial, basis_emb_size,
                               bias=False)
        self.lin_sbf2 = Linear(basis_emb_size, int_emb_size, bias=False)

        # Hidden transformation of input message:
        self.lin_kj = Linear(hidden_channels, hidden_channels)
        self.lin_ji = Linear(hidden_channels, hidden_channels)

        # Embedding projections for interaction triplets:
        self.lin_down = Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = Linear(int_emb_size, hidden_channels, bias=False)

        # Residual layers before and after skip connection:
        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
        ])
        self.lin = Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def forward(self, x, rbf, sbf, idx_kj,
                idx_ji):
        # Initial transformation:
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))

        # Transformation via Bessel basis:
        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        # Down project embedding and generating triple-interactions:
        x_kj = self.act(self.lin_down(x_kj))

        # Transform via 2D spherical basis:
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        # Aggregate interactions and up-project embeddings:
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0), reduce='sum')
        x_kj = self.act(self.lin_up(x_kj))

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h

class OutputPPBlock(torch.nn.Module):
    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        out_emb_channels: int,
        out_channels: int,
        num_layers: int,
        act: Callable,
        output_initializer: str = 'zeros',
    ):
        assert output_initializer in {'zeros', 'glorot_orthogonal'}

        super().__init__()

        self.act = act
        self.output_initializer = output_initializer

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)

        # The up-projection layer:
        self.lin_up = Linear(hidden_channels, out_emb_channels, bias=False)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(out_emb_channels, out_emb_channels))
        self.lin = Linear(out_emb_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        if self.output_initializer == 'zeros':
            self.lin.weight.data.fill_(0)
        elif self.output_initializer == 'glorot_orthogonal':
            glorot_orthogonal(self.lin.weight, scale=2.0)

    def forward(self, x: Tensor, rbf: Tensor, i: Tensor,
                num_nodes: Optional[int] = None) -> Tensor:
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes, reduce='sum')
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


def triplets(
    edge_index: Tensor,
    num_nodes: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    row, col = edge_index  # j->i

    value = torch.arange(row.size(0), device=row.device)
    adj_t = SparseTensor(row=col, col=row, value=value,
                         sparse_sizes=(num_nodes, num_nodes))
    adj_t_row = adj_t[row]
    num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

    # Node indices (k->j->i) for triplets.
    idx_i = col.repeat_interleave(num_triplets)
    idx_j = row.repeat_interleave(num_triplets)
    idx_k = adj_t_row.storage.col()
    mask = idx_i != idx_k  # Remove i == k triplets.
    idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

    # Edge indices (k-j, j->i) for triplets.
    idx_kj = adj_t_row.storage.value()[mask]
    idx_ji = adj_t_row.storage.row()[mask]

    return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji



class DimeNetPlusPlus(DimeNet):
    r"""The DimeNet++ from the `"Fast and Uncertainty-Aware
    Directional Message Passing for Non-Equilibrium Molecules"
    <https://arxiv.org/abs/2011.14115>`_ paper.

    :class:`DimeNetPlusPlus` is an upgrade to the :class:`DimeNet` model with
    8x faster and 10% more accurate than :class:`DimeNet`.

    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        int_emb_size (int): Size of embedding in the interaction block.
        basis_emb_size (int): Size of basis embedding in the interaction block.
        out_emb_channels (int): Size of embedding in the output block.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        cutoff: (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip: (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip: (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers: (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act: (str or Callable, optional): The activation funtion.
            (default: :obj:`"swish"`)
        output_initializer (str, optional): The initialization method for the
            output layer (:obj:`"zeros"`, :obj:`"glorot_orthogonal"`).
            (default: :obj:`"zeros"`)
    """

    url = ('https://raw.githubusercontent.com/gasteigerjo/dimenet/'
           'master/pretrained/dimenet_pp')

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_blocks: int,
        int_emb_size: int,
        basis_emb_size: int,
        out_emb_channels: int,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: Union[str, Callable] = 'swish',
        output_initializer: str = 'zeros',
    ):
        act = activation_resolver(act)

        super().__init__(
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            num_bilinear=1,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            envelope_exponent=envelope_exponent,
            num_before_skip=num_before_skip,
            num_after_skip=num_after_skip,
            num_output_layers=num_output_layers,
            act=act,
            output_initializer=output_initializer,
        )

        # We are re-using the RBF, SBF and embedding layers of `DimeNet` and
        # redefine output_block and interaction_block in DimeNet++.
        # Hence, it is to be noted that in the above initalization, the
        # variable `num_bilinear` does not have any purpose as it is used
        # solely in the `OutputBlock` of DimeNet:
        self.output_blocks = torch.nn.ModuleList([
            OutputPPBlock(
                num_radial,
                hidden_channels,
                out_emb_channels,
                out_channels,
                num_output_layers,
                act,
                output_initializer,
            ) for _ in range(num_blocks + 1)
        ])

        self.interaction_blocks = torch.nn.ModuleList([
            InteractionPPBlock(
                hidden_channels,
                int_emb_size,
                basis_emb_size,
                num_spherical,
                num_radial,
                num_before_skip,
                num_after_skip,
                act,
            ) for _ in range(num_blocks)
        ])

        self.reset_parameters()

[docs]    @classmethod
    def from_qm9_pretrained(
        cls,
        root: str,
        dataset: Dataset,
        target: int,
    ) -> Tuple['DimeNetPlusPlus', Dataset, Dataset,
               Dataset]:  # pragma: no cover
        r"""Returns a pre-trained :class:`DimeNetPlusPlus` model on the
        :class:`~torch_geometric.datasets.QM9` dataset, trained on the
        specified target :obj:`target`."""
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf

        assert target >= 0 and target <= 12 and not target == 4

        root = osp.expanduser(osp.normpath(root))
        path = osp.join(root, 'pretrained_dimenet_pp', qm9_target_dict[target])

        makedirs(path)
        url = f'{cls.url}/{qm9_target_dict[target]}'

        if not osp.exists(osp.join(path, 'checkpoint')):
            download_url(f'{url}/checkpoint', path)
            download_url(f'{url}/ckpt.data-00000-of-00002', path)
            download_url(f'{url}/ckpt.data-00001-of-00002', path)
            download_url(f'{url}/ckpt.index', path)

        path = osp.join(path, 'ckpt')
        reader = tf.train.load_checkpoint(path)

        # Configuration from DimeNet++:
        # https://github.com/gasteigerjo/dimenet/blob/master/config_pp.yaml
        model = cls(
            hidden_channels=128,
            out_channels=1,
            num_blocks=4,
            int_emb_size=64,
            basis_emb_size=8,
            out_emb_channels=256,
            num_spherical=7,
            num_radial=6,
            cutoff=5.0,
            max_num_neighbors=32,
            envelope_exponent=5,
            num_before_skip=1,
            num_after_skip=2,
            num_output_layers=3,
        )

        def copy_(src, name, transpose=False):
            init = reader.get_tensor(f'{name}/.ATTRIBUTES/VARIABLE_VALUE')
            init = torch.from_numpy(init)
            if name[-6:] == 'kernel':
                init = init.t()
            src.data.copy_(init)

        copy_(model.rbf.freq, 'rbf_layer/frequencies')
        copy_(model.emb.emb.weight, 'emb_block/embeddings')
        copy_(model.emb.lin_rbf.weight, 'emb_block/dense_rbf/kernel')
        copy_(model.emb.lin_rbf.bias, 'emb_block/dense_rbf/bias')
        copy_(model.emb.lin.weight, 'emb_block/dense/kernel')
        copy_(model.emb.lin.bias, 'emb_block/dense/bias')

        for i, block in enumerate(model.output_blocks):
            copy_(block.lin_rbf.weight, f'output_blocks/{i}/dense_rbf/kernel')
            copy_(block.lin_up.weight,
                  f'output_blocks/{i}/up_projection/kernel')
            for j, lin in enumerate(block.lins):
                copy_(lin.weight, f'output_blocks/{i}/dense_layers/{j}/kernel')
                copy_(lin.bias, f'output_blocks/{i}/dense_layers/{j}/bias')
            copy_(block.lin.weight, f'output_blocks/{i}/dense_final/kernel')

        for i, block in enumerate(model.interaction_blocks):
            copy_(block.lin_rbf1.weight, f'int_blocks/{i}/dense_rbf1/kernel')
            copy_(block.lin_rbf2.weight, f'int_blocks/{i}/dense_rbf2/kernel')
            copy_(block.lin_sbf1.weight, f'int_blocks/{i}/dense_sbf1/kernel')
            copy_(block.lin_sbf2.weight, f'int_blocks/{i}/dense_sbf2/kernel')

            copy_(block.lin_ji.weight, f'int_blocks/{i}/dense_ji/kernel')
            copy_(block.lin_ji.bias, f'int_blocks/{i}/dense_ji/bias')
            copy_(block.lin_kj.weight, f'int_blocks/{i}/dense_kj/kernel')
            copy_(block.lin_kj.bias, f'int_blocks/{i}/dense_kj/bias')

            copy_(block.lin_down.weight,
                  f'int_blocks/{i}/down_projection/kernel')
            copy_(block.lin_up.weight, f'int_blocks/{i}/up_projection/kernel')

            for j, layer in enumerate(block.layers_before_skip):
                copy_(layer.lin1.weight,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_1/kernel')
                copy_(layer.lin1.bias,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_1/bias')
                copy_(layer.lin2.weight,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_2/kernel')
                copy_(layer.lin2.bias,
                      f'int_blocks/{i}/layers_before_skip/{j}/dense_2/bias')

            copy_(block.lin.weight, f'int_blocks/{i}/final_before_skip/kernel')
            copy_(block.lin.bias, f'int_blocks/{i}/final_before_skip/bias')

            for j, layer in enumerate(block.layers_after_skip):
                copy_(layer.lin1.weight,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_1/kernel')
                copy_(layer.lin1.bias,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_1/bias')
                copy_(layer.lin2.weight,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_2/kernel')
                copy_(layer.lin2.bias,
                      f'int_blocks/{i}/layers_after_skip/{j}/dense_2/bias')

        random_state = np.random.RandomState(seed=42)
        perm = torch.from_numpy(random_state.permutation(np.arange(130831)))
        perm = perm.long()
        train_idx = perm[:110000]
        val_idx = perm[110000:120000]
        test_idx = perm[120000:]

        return model, (dataset[train_idx], dataset[val_idx], dataset[test_idx])

