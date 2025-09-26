import torch
import torch.nn as nn
import loss
from model.BaseModel import BaseModel
from model.MyModel.DyGCN import DyGCN
from model.MyModel.collector import Collector
from model.MyModel.decoder import Decoder
from model.MyModel.encoder import StationEncoder, DateEncoder, EnvEncoder
collectors = Collector('saved')
import torch
import torch.nn as nn
import torch.nn.functional as F
class CrossAttnFusion(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 1, dropout: float = 0.05, ff_mult: int = 2):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )

        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 可学习门控 α ∈ (0,1)，初始偏向保守（更多 residual）
        self.gate = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, station: torch.Tensor, date: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
        B, N, D = station.shape

        # Pre-Norm
        s = self.norm_q(station)
        d = self.norm_k(date)
        e = self.norm_v(env)

        q = self.q_proj(s).unsqueeze(2)                   # [B,N,1,D]
        k = torch.stack([self.k_proj(d), self.k_proj(e)], dim=2)  # [B,N,2,D]
        v = torch.stack([self.v_proj(d), self.v_proj(e)], dim=2)  # [B,N,2,D]

        BN = B * N
        q = q.reshape(BN, 1, D)
        k = k.reshape(BN, 2, D)
        v = v.reshape(BN, 2, D)

        ctx, _ = self.attn(q, k, v, need_weights=False)   # [BN,1,D]
        ctx = ctx.reshape(B, N, D)

        # 残差块
        cross_out = self.norm1(station + self.drop(ctx))
        cross_out = self.norm2(cross_out + self.drop(self.ff(cross_out)))

        # 保底 residual（简单且稳）
        residual_sum = 0.5 * (station + 0.5 * (date + env))

        # 门控混合：α 越小越靠 residual，越大越靠 cross
        alpha = self.gate(torch.cat([station, date, env], dim=-1))  # [B,N,1]
        union = (1 - alpha) * residual_sum + alpha * cross_out
        return union


class ReDyNet(BaseModel):
    def __init__(self, config, data_feature):
        super().__init__(data_feature)
        self.total_dim = data_feature['feature_dim']
        self.output_dim = data_feature.get('output_dim')
        self.input_time = config['input_time']
        self.output_time = config['output_time']
        self.node_num = data_feature['num_nodes']

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self._scaler = data_feature.get('scaler')

        self.ext_dim = data_feature.get('ext_dim')
        self.gcn_dim = config['gcn_dim']
        self.encoder_dim = config['encoder_dim']
        self.station_embed_dim = config['station_embed_dim']
        self.date_embed_dim = config['date_embed_dim']
        self.vae_dim = config['vae_dim']

        self.cheby_k = config['cheby_k']
        self.a = config['a']
        self.temperature = config['temperature']
        self.beta = config['beta']
        self.vae_ratio = config['vae_ratio']

        self.station_encoder = StationEncoder(num_sites=self.node_num, encoder_dim=self.encoder_dim,
                                              station_embed_dim=self.station_embed_dim,
                                              input_time=self.input_time)
        self.date_encoder = DateEncoder(encoder_dim=self.encoder_dim, date_embed_dim=self.date_embed_dim,
                                        input_time=self.input_time)
        self.env_encoder = EnvEncoder(encoder_dim=self.encoder_dim, input_time=self.input_time)

        self.dygcn = DyGCN(dim_in=self.input_time * self.output_dim, dim_out=self.gcn_dim, cheby_k=self.cheby_k,
                           embed_dim=self.encoder_dim)

        self.decoder = Decoder(input_dim=self.gcn_dim, output_dim=self.output_dim * self.output_time)

        self.gcn_activation = nn.GELU()

        self.union_norm = nn.LayerNorm(self.encoder_dim)
        self.station_norm = nn.LayerNorm(self.encoder_dim)
        self.date_norm = nn.LayerNorm(self.encoder_dim)
        self.env_norm = nn.LayerNorm(self.encoder_dim)
        self.fusion = CrossAttnFusion(d_model=self.encoder_dim, n_heads=1, dropout=0.05, ff_mult=2)


        self.rec_layer_norm = nn.LayerNorm(self.encoder_dim)

        self._init_parameters()

    def forward(self, batch, return_supports=False):
        x = batch['X']  # [B,T,N,C]

        flow = x[..., :self.output_dim]  # [B,T,N,2]
        time = x[..., self.output_dim:self.output_dim + 2]  # [B,T,N,2]
        env = x[..., self.output_dim + 2:]  # [B,T,N,5]

        flow_station = self.station_encoder(flow)  # [B,N,64]
        flow_date = self.date_encoder(flow, time)  # [B,N,64]
        flow_env = self.env_encoder(flow, env)  # [B,N,64]


        union =  flow_station
        union = self.union_norm(union)

        flow_station = self.station_norm(flow_station)
        flow_date = self.date_norm(flow_date)
        flow_env = self.env_norm(flow_env)

        flow = flow.permute(0, 2, 1, 3)  # [B,N,T,C]
        flow = flow.reshape(flow.shape[0], flow.shape[1], -1)

        gcn_output = self.dygcn(flow, union, flow_station, return_supports=False)
        gcn_output = self.gcn_activation(gcn_output)

        output = self.decoder(gcn_output)
        output = output.reshape(output.shape[0], output.shape[1], self.output_time, self.output_dim)
        output = output.permute(0, 2, 1, 3)

        return output, flow_station, flow_date, flow_env

    def calculate_loss_train(self, batch):
        y_true = batch['y']
        y_predicted, flow_station, flow_date, flow_env= self.forward(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        loss_pred = loss.huber_loss_torch(y_predicted, y_true)
        return loss_pred

    def predict(self, batch):
        output, _, _, _= self.forward(batch)
        return output

    def reparameterize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        eps = std.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def _init_parameters(self):
        print('Initializing parameters...')
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
