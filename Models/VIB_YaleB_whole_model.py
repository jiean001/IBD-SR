from abc import ABC
import torch
import torch.nn.init as init
import torch.nn.modules as nn
from Models.Leyers import Conv2DSamePadding, Reshape
from torch.nn import Parameter

eps = 1e-6
use_cuda = True


class VIB_Encoder(nn.Module, ABC):
    def __init__(self, shape_data, dim_embedding_clean, dim_embedding_noise):
        super(VIB_Encoder, self).__init__()
        self.dim_data = shape_data

        self.mu_encoder_clean = nn.Sequential(
            nn.Linear(self.dim_data, dim_embedding_clean),
            nn.BatchNorm1d(num_features=dim_embedding_clean),
            nn.Tanh()
        )
        self.log_sigma_encoder_clean = nn.Linear(self.dim_data, dim_embedding_clean)
        self.log_pi_encoder_clean = nn.Linear(self.dim_data, dim_embedding_clean)

        self.mu_encoder_noise = nn.Sequential(
            nn.Linear(self.dim_data, dim_embedding_noise),
            nn.BatchNorm1d(num_features=dim_embedding_noise),
            nn.Tanh()
        )
        self.log_sigma_encoder_noise = nn.Linear(self.dim_data, dim_embedding_noise)
        self.log_pi_encoder_noise = nn.Linear(self.dim_data, dim_embedding_noise)

    def forward(self, input_data):
        mu_clean = self.mu_encoder_clean(input_data)
        log_sigma_clean = self.log_sigma_encoder_clean(input_data)
        log_pi_clean = self.log_pi_encoder_clean(input_data)
        gamma_clean = torch.sigmoid(log_pi_clean)

        mu_noise = self.mu_encoder_noise(input_data)
        log_sigma_noise = self.log_sigma_encoder_noise(input_data)
        log_pi_noise = self.log_pi_encoder_noise(input_data)
        gamma_noise = torch.sigmoid(log_pi_noise)

        return (mu_clean, log_sigma_clean, log_pi_clean, gamma_clean,
                mu_noise, log_sigma_noise, log_pi_noise, gamma_noise)


class VIB_Decoder(nn.Module, ABC):
    def __init__(self, shape_data, dim_embedding):
        super(VIB_Decoder, self).__init__()
        self.dim_data = shape_data
        self.decoder = nn.Sequential(
            nn.Linear(dim_embedding, self.dim_data),
            nn.Tanh()
        )

    def forward(self, input_data_clean, input_data_noise):
        input_data = torch.cat((input_data_clean, input_data_noise), dim=1)
        reconstructed_data = self.decoder(input_data)
        return reconstructed_data


class VIB_Classifier(nn.Module, ABC):
    def __init__(self, dim_embedding, num_target_class):
        super(VIB_Classifier, self).__init__()

        self.decoder = nn.Sequential(
            nn.BatchNorm1d(dim_embedding),
            nn.Linear(dim_embedding, num_target_class),
        )

    def forward(self, input_data):
        classification_logit = self.decoder(input_data)
        return classification_logit


class VariationalInformationBottleneck(nn.Module, ABC):
    def __init__(self, shape_data, num_target_class, dim_embedding_clean, dim_embedding_noise, parameter):
        super(VariationalInformationBottleneck, self).__init__()

        self.reconstruction_weight = parameter["reconstruction_weight"]
        self.pairwise_kl_clean_weight = parameter["pairwise_kl_clean_weight"]
        self.pairwise_kl_noise_weight = parameter["pairwise_kl_noise_weight"]
        self.sparse_kl_weight_clean = parameter["sparse_kl_weight_clean"]
        self.sparse_kl_weight_noise = parameter["sparse_kl_weight_noise"]

        self.sparsity_clean = parameter["sparsity_clean"]
        self.sparsity_noise = parameter["sparsity_noise"]

        num_sensitive_class = parameter["num_sensitive_class"]

        self.encoder = VIB_Encoder(shape_data, dim_embedding_clean, dim_embedding_noise)
        self.decoder = VIB_Decoder(shape_data, dim_embedding_clean + dim_embedding_noise)
        self.classifier = VIB_Classifier(dim_embedding_clean, num_target_class)
        self.classifier_noise = VIB_Classifier(dim_embedding_noise, num_sensitive_class)

    @staticmethod
    def reparameterize(mu, log_sigma, log_pi, sigmoid_temperature, num_samples):
        batch_size = mu.shape[0]
        dim_embedding = mu.shape[1]
        mu_repeat = mu.unsqueeze(dim=1).repeat(1, num_samples, 1)
        log_sigma_repeat = log_sigma.unsqueeze(dim=1).repeat(1, num_samples, 1)
        log_pi_repeat = log_pi.unsqueeze(dim=1).repeat(1, num_samples, 1)

        std = torch.exp(0.5 * log_sigma_repeat)
        noise_eps = torch.randn_like(std)
        noise_eta = torch.rand_like(std)
        if use_cuda:
            noise_eps = noise_eps.cuda()
            noise_eta = noise_eta.cuda()

        noise_eta = torch.clamp(noise_eta, min=eps, max=1 - eps)
        gumbel_U = torch.log(noise_eta) - torch.log(1 - noise_eta)
        spike_logit = log_pi_repeat + gumbel_U
        spike_prob = torch.sigmoid(spike_logit * sigmoid_temperature)

        gaussian = noise_eps.mul(std).add(mu_repeat)
        output_sample = spike_prob.mul(gaussian)
        output_sample_rearrange = output_sample.view(batch_size * num_samples, dim_embedding)
        return output_sample_rearrange, output_sample

    @staticmethod
    def pairwise_kl_loss(mu, log_sigma, gamma, batch_size):
        gamma = torch.clamp(gamma, min=eps, max=1 - eps)

        mu1 = mu.unsqueeze(dim=1).repeat(1, batch_size, 1)
        log_sigma1 = log_sigma.unsqueeze(dim=1).repeat(1, batch_size, 1)
        gamma1 = gamma.unsqueeze(dim=1).repeat(1, batch_size, 1)

        mu2 = mu.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        log_sigma2 = log_sigma.unsqueeze(dim=0).repeat(batch_size, 1, 1)
        gamma2 = gamma.unsqueeze(dim=0).repeat(batch_size, 1, 1)

        kl_divergence1 = 0.5 * (log_sigma2 - log_sigma1)
        kl_divergence2 = 0.5 * torch.div(torch.exp(log_sigma1) + torch.square(mu1 - mu2), torch.exp(log_sigma2))
        kl_divergence_loss1 = torch.mul(gamma1, kl_divergence1 + kl_divergence2 - 0.5)

        kl_divergence3 = (1 - gamma1).mul(torch.log(1 - gamma1) - torch.log(1 - gamma2))
        kl_divergence4 = gamma1.mul(torch.log(gamma1) - torch.log(gamma2))
        kl_divergence_loss2 = kl_divergence3 + kl_divergence4

        pairwise_kl_divergence_loss = (kl_divergence_loss1 + kl_divergence_loss2).sum(-1).sum(-1) / (batch_size - 1)

        return pairwise_kl_divergence_loss

    @staticmethod
    def sparse_kl_loss(gamma, sparsity):
        sparsity = torch.ones_like(gamma) * sparsity
        sparsity = torch.clamp(sparsity, min=eps, max=1 - eps)
        sparsity_mean = sparsity.mean(-1)

        gamma = torch.clamp(gamma, min=eps, max=1 - eps)
        gamma_mean = gamma.mean(-1)

        kl_divergence1 = (1 - gamma_mean).mul(torch.log(1 - gamma_mean) - torch.log(1 - sparsity_mean))
        kl_divergence2 = gamma_mean.mul(torch.log(gamma_mean) - torch.log(sparsity_mean))
        sparse_kl_divergence_loss = gamma.shape[1] * (kl_divergence1 + kl_divergence2)

        return sparse_kl_divergence_loss

    def forward(self, input_data, input_label, input_sensitive_labels=None, num_samples=None, training=None, drawing=False):
        batch_size = input_data.shape[0]
        label_target = input_label.unsqueeze(dim=1).repeat(1, num_samples)
        label_target = label_target.view(batch_size * num_samples)
        sensitive_label_target = None
        if input_sensitive_labels is not None:
            sensitive_label_target = input_sensitive_labels.unsqueeze(dim=1).repeat(1, num_samples)
            sensitive_label_target = sensitive_label_target.view(batch_size * num_samples)

        data_target = input_data.unsqueeze(dim=1).repeat(1, num_samples, 1)
        data_target = data_target.view(batch_size * num_samples, -1)

        (mu_clean, log_sigma_clean, log_pi_clean, gamma_clean,
         mu_noise, log_sigma_noise, log_pi_noise, gamma_noise) = self.encoder(input_data)

        data_embedding_clean, output_data_embedding_clean = self.reparameterize(
            mu_clean, log_sigma_clean, log_pi_clean, sigmoid_temperature=1.0, num_samples=num_samples
        )
        data_embedding_noise, output_data_embedding_noise = self.reparameterize(
            mu_noise, log_sigma_noise, log_pi_noise, sigmoid_temperature=1.0, num_samples=num_samples
        )

        if training:
            # Classification Loss
            classification_logit = self.classifier(data_embedding_clean)
            classification_loss = nn.CrossEntropyLoss(reduction='none')(
                input=classification_logit, target=label_target
            )
            classification_loss = classification_loss.view(batch_size, num_samples).mean(-1)

            classification_sensitive_logit = self.classifier_noise(data_embedding_noise)
            classification_sensitive_loss = nn.CrossEntropyLoss(reduction='none')(
                input=classification_sensitive_logit, target=sensitive_label_target
            )
            classification_sensitive_loss = classification_sensitive_loss.view(batch_size, num_samples).mean(-1)

            # # Reconstruction Loss
            reconstruction_data = self.decoder(data_embedding_clean, data_embedding_noise)
            reconstruction_data = reconstruction_data.view(batch_size * num_samples, -1)
            reconstruction_loss = torch.square(data_target - reconstruction_data).mean(-1)
            reconstruction_loss = reconstruction_loss.view(batch_size, num_samples).mean(-1)

            # Clean Pairwise KL Divergence Loss
            pairwise_kl_loss_clean = self.pairwise_kl_loss(mu_clean, log_sigma_clean, gamma_clean, batch_size)

            # Noise Pairwise KL Divergence Loss
            pairwise_kl_loss_noise = self.pairwise_kl_loss(mu_noise, log_sigma_noise, gamma_noise, batch_size)

            # Clean Sparse KL Divergence Loss
            sparse_kl_loss_clean = self.sparse_kl_loss(gamma_clean, self.sparsity_clean)

            # Noise Sparse KL Divergence Loss
            sparse_kl_loss_noise = self.sparse_kl_loss(gamma_noise, self.sparsity_noise)

            total_loss = (classification_loss + classification_sensitive_loss
                          + self.reconstruction_weight * reconstruction_loss
                          + self.pairwise_kl_clean_weight * pairwise_kl_loss_clean
                          + self.pairwise_kl_noise_weight * pairwise_kl_loss_noise
                          + self.sparse_kl_weight_clean * sparse_kl_loss_clean
                          + self.sparse_kl_weight_noise * sparse_kl_loss_noise)

            return (total_loss, classification_loss, classification_sensitive_loss, reconstruction_loss,
                    pairwise_kl_loss_clean, pairwise_kl_loss_noise,
                    sparse_kl_loss_clean, sparse_kl_loss_noise)

        elif drawing:
            classification_logit = self.classifier(data_embedding_clean)
            classification_prob = torch.softmax(classification_logit, dim=-1).view(batch_size, num_samples, -1).mean(1)
            return output_data_embedding_clean, output_data_embedding_noise, classification_prob
        else:
            classification_logit = self.classifier(data_embedding_clean)
            classification_prob = torch.softmax(classification_logit, dim=-1).view(batch_size, num_samples, -1).mean(1)
            return output_data_embedding_clean, classification_prob


class VIB_Discriminator(nn.Module, ABC):
    def __init__(self, num_sensitive_class, dim_embedding):
        super(VIB_Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(dim_embedding, num_sensitive_class),
        )

    def forward(self, input_data, input_target):
        batch_size = input_data.shape[0]
        num_samples = input_data.shape[1]
        dim_embedding = input_data.shape[2]
        input_data_rearrange = input_data.view(batch_size * num_samples, dim_embedding)
        input_target = input_target.unsqueeze(dim=1).repeat(1, num_samples)
        input_target = input_target.view(batch_size * num_samples)

        discriminator_logit = self.discriminator(input_data_rearrange)
        discriminator_loss = nn.CrossEntropyLoss(reduction='none')(discriminator_logit, input_target)
        discriminator_loss = discriminator_loss.view(batch_size, num_samples).mean(-1)

        discriminator_prob = torch.softmax(discriminator_logit, dim=-1).view(batch_size, num_samples, -1).mean(1)

        return discriminator_loss, discriminator_prob
