
import torch
import torch.nn as nn
import torch.nn.functional as F

def F_interpolate_1d(x: torch.Tensor, target_len: int) -> torch.Tensor:
    B, C, N, Flen = x.shape
    return F.interpolate(x, size=(N, target_len), mode="bilinear", align_corners=False)

class AdvancedSpectralSupport(nn.Module):
    def __init__(self, n_bands=3, ms_fft_scales=(1, 2, 4),
                 freq_conv_channels=32, use_freq_conv=True,
                 use_freq_attn=True, fusion_mode="softmax",
                 return_all_graphs=False):
        super().__init__()
        self.n_bands = n_bands
        self.ms_fft_scales = ms_fft_scales
        self.use_freq_conv = use_freq_conv
        self.use_freq_attn = use_freq_attn
        self.return_all_graphs = return_all_graphs
        self._deferred_inited = not use_freq_conv
        self._freq_out_ch = freq_conv_channels

        self.register_parameter("fusion_logits", nn.Parameter(torch.zeros(8)))
        self._fusion_size = None

    @staticmethod
    def _l2norm_last(x: torch.Tensor, eps=1e-8):
        return x / (x.norm(p=2, dim=-1, keepdim=True) + eps)

    @staticmethod
    def _cosine_graph(node_feat: torch.Tensor) -> torch.Tensor:
        return torch.matmul(node_feat, node_feat.transpose(1, 2))

    @staticmethod
    def _row_softmax(A: torch.Tensor) -> torch.Tensor:
        return F.softmax(A, dim=-1)

    @staticmethod
    def _split_bands(x: torch.Tensor, n_bands: int):
        B, N, F = x.shape
        base, rem, start = F // n_bands, F % n_bands, 0
        bands = []
        for b in range(n_bands):
            length = base + (1 if b < rem else 0)
            bands.append(x[..., start:start+length])
            start += length
        return bands

    def _interp_to(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        return F_interpolate_1d(x.unsqueeze(1), target_len).squeeze(1)

    def _ensure_freq_modules(self, in_ch: int, device=None):
        if self._deferred_inited:
            return
        self.freq_conv = nn.Conv1d(in_ch, self._freq_out_ch, kernel_size=3, padding=1, bias=False)
        self.freq_bn = nn.BatchNorm1d(self._freq_out_ch)
        self.freq_attn_proj = nn.Conv1d(self._freq_out_ch, 1, kernel_size=1, bias=True) if self.use_freq_attn else None

        device = device or next(self.parameters()).device
        self.freq_conv = self.freq_conv.to(device)
        self.freq_bn = self.freq_bn.to(device)
        if self.freq_attn_proj:
            self.freq_attn_proj = self.freq_attn_proj.to(device)

        self._deferred_inited = True

    def forward(self, all_emb: torch.Tensor):
        B, N, D = all_emb.shape
        device = all_emb.device

        X = torch.fft.rfft(all_emb, n=D, dim=-1)
        F0 = X.shape[-1]
        mag = X.abs()

        base_feat = mag.unsqueeze(2)  # [B,N,1,F0]
        C_base = 1

        # Band graphs
        mag_norm = self._l2norm_last(mag)
        bands = self._split_bands(mag_norm, self.n_bands)
        band_graphs = [self._cosine_graph(self._l2norm_last(b)) for b in bands]

        # Multi-scale FFT graphs
        ms_graphs = []
        for s in self.ms_fft_scales:
            n_len = max(4, D // s)
            Xs = torch.fft.rfft(all_emb, n=n_len, dim=-1)
            Ms = self._interp_to(Xs.abs(), F0)
            ms_graphs.append(self._cosine_graph(self._l2norm_last(Ms)))

        # Learned spectral graph
        learned_graph = None
        if self.use_freq_conv:
            self._ensure_freq_modules(C_base, device)
            z = base_feat.reshape(B * N, C_base, F0)
            z = F.gelu(self.freq_bn(self.freq_conv(z)))
            node_vec = (z * F.softmax(self.freq_attn_proj(z), dim=-1)).sum(dim=-1) if self.use_freq_attn else z.mean(dim=-1)
            node_vec = self._l2norm_last(node_vec.view(B, N, -1))
            learned_graph = self._cosine_graph(node_vec)

        # Base graph
        base_vec = self._l2norm_last(base_feat.reshape(B, N, -1))
        base_graph = self._cosine_graph(base_vec)

        graphs = [base_graph] + band_graphs + ms_graphs
        if learned_graph is not None:
            graphs.append(learned_graph)

        K = len(graphs)
        if self._fusion_size != K:
            with torch.no_grad():
                self.fusion_logits = nn.Parameter(torch.zeros(K, device=device))
            self._fusion_size = K

        alphas = F.softmax(self.fusion_logits, dim=0)
        S = sum(w * G for w, G in zip(alphas, graphs))
        supports = self._row_softmax(S)

        return (supports, graphs, alphas) if self.return_all_graphs else supports

class DyGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheby_k, embed_dim, aggregate_type='sum'):
        super().__init__()
        self.cheby_k = cheby_k
        self.aggregate_type = aggregate_type
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheby_k, dim_in, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        if aggregate_type == 'weighted_sum':
            self.weights_cheby = nn.Parameter(torch.ones(cheby_k))
        self.spectral_builder = AdvancedSpectralSupport()

    def forward(self, x, all_emb, station_emb, return_supports=False):
        B, N, _ = all_emb.shape
        supports = self.spectral_builder(all_emb)

        t_k_0 = torch.eye(N, device=supports.device).unsqueeze(0).expand(B, -1, -1)
        support_set = [t_k_0, supports]
        for k in range(2, self.cheby_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports_cheby = torch.stack(support_set).permute(1, 0, 2, 3)

        weights = torch.einsum('bni,ikop->bnkop', station_emb, self.weights_pool)
        bias = torch.matmul(station_emb, self.bias_pool)
        x_g = torch.einsum('bkij,bjd->bkid', supports_cheby, x)
        x_g_conv = torch.einsum('bkni,bnkio->bnko', x_g, weights)

        x_g_conv = x_g_conv.sum(dim=2) + bias if self.aggregate_type == 'sum' else (x_g_conv * self.weights_cheby[None, None, :, None]).sum(dim=2) + bias

        return (x_g_conv, supports) if return_supports else x_g_conv


