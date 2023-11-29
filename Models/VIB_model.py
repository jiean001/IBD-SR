from abc import ABC
import numpy as np
import torch
from torch.nn import Parameter
import torch.nn.init as init
import torch.nn.modules as nn
import torch.nn.functional as F

SMALL_CONSTANT = 1e-6


class VIB_Encoder(nn.Module, ABC):
    def __init__(self, dim_data, dim_embedding, dim_hidden_encoder):
        super(VIB_Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(dim_data, dim_hidden_encoder),
            nn.ReLU(),
            nn.Linear(dim_hidden_encoder, dim_hidden_encoder),
            nn.ReLU()
        )

        self.mu_encoder = nn.Linear(dim_hidden_encoder, dim_embedding)
        self.log_sigma_encoder = nn.Linear(dim_hidden_encoder, dim_embedding)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight.data)
                init.constant_(m.bias.data, 0)

    def forward(self, input_data):
        hidden_feature = self.encoder(input_data)
        embedding_mu = self.mu_encoder(hidden_feature)
        embedding_log_sigma = 2 * torch.log(F.softplus(self.log_sigma_encoder(hidden_feature) - 5, beta=1))
        return embedding_mu, embedding_log_sigma


class VIB_Decoder(nn.Module, ABC):
    def __init__(self, dim_embedding, num_target_class):
        super(VIB_Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(dim_embedding, num_target_class),
            nn.Softmax(dim=-1)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight.data)
                init.constant_(m.bias.data, 0)

    def forward(self, input_data):
        classification_prob = self.decoder(input_data)
        return classification_prob


class VariationalInformationBottleneck(nn.Module, ABC):
    def __init__(self, dim_data, num_target_class, dim_embedding, dim_hidden_encoder, alpha):
        super(VariationalInformationBottleneck, self).__init__()

        self.alpha = alpha
        self.encoder = VIB_Encoder(dim_data, dim_embedding, dim_hidden_encoder)
        self.decoder = VIB_Decoder(dim_embedding, num_target_class)

    @staticmethod
    def reparameterize(mu, log_sigma, num_samples):
        batch_size = mu.shape[0]
        dim_embedding = mu.shape[1]
        mu_repeat = mu.unsqueeze(dim=1).repeat(1, num_samples, 1)
        log_sigma_repeat = log_sigma.unsqueeze(dim=1).repeat(1, num_samples, 1)
        std = torch.exp(0.5 * log_sigma_repeat)
        noise_eps = torch.randn_like(std)
        gaussian_sample = noise_eps.mul(std).add(mu_repeat)
        gaussian_sample = gaussian_sample.view(batch_size * num_samples, dim_embedding)
        return gaussian_sample

    @staticmethod
    def kl_divergence_loss(mu, log_sigma):
        std = torch.exp(0.5 * log_sigma)
        kl_divergence_loss = 0.5 * (mu.pow(2) + torch.exp(log_sigma) - log_sigma - 1).sum(-1)
        kl_divergence_loss2 = -0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(-1)
        return kl_divergence_loss

    def forward(self, input_data, input_target, num_samples):
        batch_size = input_data.shape[0]
        input_data = input_data.view(batch_size, -1)
        embedding_mu, embedding_log_sigma = self.encoder(input_data)
        data_embedding = self.reparameterize(embedding_mu, embedding_log_sigma, num_samples)
        classification_prob = self.decoder(data_embedding)
        classification_prob_mean = classification_prob.view(batch_size, num_samples, -1).mean(1)
        classification_loss = nn.NLLLoss(reduction='none')(torch.log(classification_prob_mean), input_target)
        kl_divergence_loss = self.kl_divergence_loss(embedding_mu, embedding_log_sigma)
        total_loss = classification_loss + self.alpha * kl_divergence_loss
        return total_loss, classification_loss, kl_divergence_loss, classification_prob_mean


class Weight_EMA(object):

    def __init__(self, model, initial_state_dict, decay=0.999):
        self.model = model
        self.model.load_state_dict(initial_state_dict, strict=True)
        self.decay = decay

    def update(self, new_state_dict):
        state_dict = self.model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = self.decay * state_dict[key] + (1 - self.decay) * new_state_dict[key]
        self.model.load_state_dict(state_dict)
