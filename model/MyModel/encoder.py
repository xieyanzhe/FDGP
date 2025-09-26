class StationEncoder(nn.Module):
    def __init__(self, num_sites, encoder_dim, station_embed_dim, input_time):
        super(StationEncoder, self).__init__()
        self.input_time = input_time
        self.site_embedding = nn.Embedding(num_sites, station_embed_dim)
        self.flow_encoder = nn.Linear(input_time * 2, encoder_dim)
        self.embed_encoder = nn.Linear(station_embed_dim, encoder_dim)
        self.dropout = nn.Dropout(0.1)
        self.encoder_dim = encoder_dim

    def forward(self, flow):
        B, T, N, C = flow.shape

        site_indices = torch.arange(N).to(flow.device)
        site_indices = site_indices.unsqueeze(0).expand(B, N)
        site_embeds = self.site_embedding(site_indices)
        site_embeds = self.dropout(site_embeds)

        flow = flow.permute(0, 2, 1, 3).reshape(B, N, T * C)
        flow = self.flow_encoder(flow)
        site_features = self.embed_encoder(site_embeds)

        p = torch.sigmoid(site_features)
        q = torch.tanh(flow)

        out = p * q + (1 - p) * site_features
        return out.view(B, N, self.encoder_dim)


class DateEncoder(nn.Module):
    def __init__(self, encoder_dim, date_embed_dim, input_time):
        super(DateEncoder, self).__init__()
        self.input_time = input_time
        self.hour_embedding = nn.Embedding(24, date_embed_dim)
        self.weekday_embedding = nn.Embedding(7, date_embed_dim)
        self.flow_encoder = nn.Linear(input_time * 2, encoder_dim)
        self.embed_encoder = nn.Linear(input_time * date_embed_dim, encoder_dim)
        self.dropout = nn.Dropout(0.1)
        self.date_embed_dim = date_embed_dim
        self.encoder_dim = encoder_dim

    def forward(self, flow, date):
        B, T, N, C = flow.shape
        hour = date[..., 0].long()
        weekday = date[..., 1].long()

        hour_embeds = self.hour_embedding(hour)
        weekday_embeds = self.weekday_embedding(weekday)
        date_embeds = hour_embeds + weekday_embeds
        date_embeds = date_embeds.permute(0, 2, 1, 3).reshape(B, N, T * self.date_embed_dim)
        date_embeds = self.dropout(date_embeds)

        flow = flow.permute(0, 2, 1, 3).reshape(B, N, T * C)
        flow = self.flow_encoder(flow)
        date_feature = self.embed_encoder(date_embeds)

        p = torch.sigmoid(date_feature)
        q = torch.tanh(flow)

        out = p * q + (1 - p) * date_feature
        return out.view(B, N, self.encoder_dim)


class EnvEncoder(nn.Module):
    def __init__(self, encoder_dim, input_time):
        super(EnvEncoder, self).__init__()
        self.input_time = input_time
        self.flow_encoder = nn.Linear(input_time * 2, encoder_dim)
        self.env_encoder = nn.Linear(input_time * 5, encoder_dim)
        self.dropout = nn.Dropout(0.1)
        self.encoder_dim = encoder_dim

    def forward(self, flow, env):
        B, T, N, C = flow.shape
        _, _, _, E = env.shape

        flow = flow.permute(0, 2, 1, 3).reshape(B, N, T * C)
        flow = self.flow_encoder(flow)

        env = env.permute(0, 2, 1, 3).reshape(B, N, T * E)
        env = self.dropout(env)
        env_feature = self.env_encoder(env)

        p = torch.sigmoid(env_feature)
        q = torch.tanh(flow)

        out = p * q + (1 - p) * env_feature
        return out.view(B, N, self.encoder_dim)
