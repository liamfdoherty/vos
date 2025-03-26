import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
from copy import deepcopy

class VOS():
    def __init__(self, backbone, ood_detector, data, queue_size = 1024, device = "cuda"):
        # Set up backbone pretrained architecture
        # TODO: Include check that the backbone has the mode compliance
        self.backbone = backbone
        self.feature_dim = backbone.finalizer.in_features

        # Set up device and move networks there
        self.device = torch.device(device)
        self.backbone = self.backbone.to(self.device)

        # Set up OOD detector architecture
        self.ood_detector = ood_detector.to(self.device)

        # Define dataset and split according to class label
        samples, targets = data.tensors[0], data.tensors[1]
        self.data_dict = {key:[] for key in targets.tolist()}
        for pair in zip(samples, targets):
            self.data_dict[int(pair[1])].append(pair[0])

        # Initialize the queue
        self.queue_size = queue_size
        self.queue = {key:random.sample(self.data_dict[key], self.queue_size) for key in targets.tolist()}

        # Initialize the OOD proposals queue. These are latent vectors, NOT original data points
        self.ood_queue = {key:[] for key in targets.tolist()}

        # Initialize the underlying statistical distributions (class-conditioned multivariate Gaussians)
        self.means = {key:torch.zeros(self.feature_dim).to(self.device) for key in self.queue.keys()}
        self.cov = torch.eye(self.feature_dim).to(self.device)
        self.cov_components = {key:torch.zeros(self.cov.shape) for key in self.queue.keys()}
        self.class_conditional_modes = {key:MultivariateNormal(self.means[key], self.cov) for key in self.queue.keys()}

    def update_queue(self, n_samples = 1):
        for target in self.queue.keys():
            del self.queue[target][:n_samples]
            new_sample = random.sample(self.data_dict[target], n_samples)
            self.queue[target].extend(new_sample)
        return None
    
    def update_gmm(self):
        for key in self.queue.keys():
            class_samples = torch.stack(self.queue[key])
            class_samples = class_samples.to(self.device)
            latent_embeddings = self.backbone(class_samples, mode = "partial").squeeze()

            # Step 1: Get per class latent representation means
            self.means[key] = torch.mean(latent_embeddings, dim = 0).detach()

            # Step 2: Get components from the inner sum of the empirical covariance
            centered_embeddings = latent_embeddings - self.means[key]
            cov_component = torch.zeros(self.cov.shape, device = self.device)
            for column in centered_embeddings:
                cov_component += torch.outer(column, column)
            cov_component = cov_component/(self.queue_size - 1)
            cov_component = cov_component.detach() + 0.001*torch.eye(self.cov.shape[0], device = self.device)
            self.cov_components[key] = cov_component

        # Step 3: Empirically estimate tied covariance
        cov = torch.zeros(self.cov.shape, device = self.device)
        for key in self.cov_components.keys():
            cov += self.cov_components[key]
        cov = cov/(len(self.cov_components))
        self.cov = cov.detach() + 0.001*torch.eye(cov.shape[0], device = self.device)

        # Step 4: Update the GMM state with computed values. Here we are using local, not tied, covariance.
        for key in self.queue.keys():
            self.class_conditional_modes[key].loc = self.means[key].to(self.device)
            self.class_conditional_modes[key].cov = self.cov_components[key].to(self.device)

        return None
    
    def compute_gmm_log_likelihood(self, latents, target):
        mode = self.class_conditional_modes[target]
        log_likelihood = mode.log_prob(latents)
        return log_likelihood

    def select_log_epsilon(self, t:int = 100): # t corresponds to the lowest ~1% of ID data likelihoods
        assert t > 0, "t must be a positive integer!"
        id_likelihoods = torch.tensor([], device = self.device)
        for key in self.queue.keys():
            id_points = torch.stack(self.queue[key])
            id_points = id_points.to(self.device)
            latents = self.backbone(id_points, mode = "partial").squeeze().detach()
            likelihoods = self.compute_gmm_log_likelihood(latents, key)
            id_likelihoods = torch.cat((id_likelihoods, likelihoods), dim = 0)
        log_epsilon = -torch.topk(-id_likelihoods, t).values[-1]
        return log_epsilon

    def sample_ood(self, max_iterations = 1000, lr = 5e-3):
        for key in self.queue.keys():
            samples = deepcopy(self.queue[key])
            samples = torch.stack(samples)
            samples = samples.to(self.device)
            latents = self.backbone(samples, mode = "partial").detach().squeeze()
            latents.requires_grad = True
            opt = torch.optim.Adam([latents], lr = lr)
            eps = self.select_log_epsilon()
            criterion = lambda x: torch.max(torch.tensor([0.], device = self.device),
                                            self.compute_gmm_log_likelihood(x, key) - eps)

            for iteration in range(max_iterations):
                opt.zero_grad()
                l = torch.mean(criterion(latents))
                l.backward()
                opt.step()
            self.ood_queue[key] = latents.detach()

        return None

    def free_energy(self, logits):
        energy_scores = -torch.logsumexp(logits, dim = 1).unsqueeze(1)
        return energy_scores

    def uncertainty_loss(self):
        id_samples = torch.Tensor([])
        for key in self.queue.keys():
            new_samples = torch.stack(self.queue[key])
            id_samples = torch.cat((id_samples, new_samples))
        id_samples = id_samples[torch.randperm(id_samples.shape[0])]
        id_samples = id_samples.to(self.device)

        id_logits = self.backbone(id_samples)

        ood_latents = self.ood_queue[key]
        classifier = list(self.backbone.children())[-1]
        ood_logits = classifier(ood_latents)

        id_energies, ood_energies = self.free_energy(id_logits), self.free_energy(ood_logits)
        id_energy_surface, ood_energy_surface = self.ood_detector(id_energies), self.ood_detector(ood_energies)

        id_loss = torch.mean(-torch.log(torch.exp(-id_energy_surface)/(1 + torch.exp(-id_energy_surface))))
        ood_loss = torch.mean(-torch.log(1/(1 + torch.exp(-ood_energy_surface))))

        return (id_loss + ood_loss)

    def train(self, iterations = 50, lr = 1e-3, beta = 0.1, ood_iterations = 3):
        # Give a for loop for the iterations, and define the necessary stuff (optimizers, etc)
        self.backbone.train()
        self.backbone.to(self.device)
        self.ood_detector.to(self.device)
        opt = torch.optim.Adam(self.backbone.parameters(), lr = lr)
        ood_opt = torch.optim.Adam(self.ood_detector.parameters(), lr = lr)
        opt_scheduler = ReduceLROnPlateau(opt, factor = 0.5)
        ood_opt_scheduler = ReduceLROnPlateau(ood_opt, factor = 0.5)
        class_criterion = nn.CrossEntropyLoss()
        for iteration in range(iterations):
            print(f"Iteration: {iteration}")
            # Update ID queue
            self.update_queue(n_samples = 512)

            # Online estimator of GMM
            self.update_gmm()

            # Update OOD queue with virtual synthesis
            self.sample_ood()

            # Given ID and OOD samples, update backbone network with combination of losses
            # TODO: Figure out logic for getting these samples and targets. Ideally, a running queue would be good.
            id_samples = torch.Tensor([])
            id_labels = torch.Tensor([])
            for key in self.queue.keys():
                new_samples = torch.stack(self.queue[key])
                new_labels = key*torch.ones(new_samples.shape[0])
                id_samples = torch.cat((id_samples, new_samples))
                id_labels = torch.cat((id_labels, new_labels)).int().long()
            idxs = torch.randperm(id_samples.shape[0])
            id_samples = id_samples[idxs]
            id_labels = id_labels[idxs]
            id_samples, id_labels = id_samples.to(self.device), id_labels.to(self.device)

            opt.zero_grad()

            outputs = self.backbone(id_samples)
            class_loss = class_criterion(outputs, F.one_hot(id_labels).float())
            uncertainty_loss = self.uncertainty_loss()

            total_loss = class_loss + beta*uncertainty_loss
            total_loss.backward()
            opt.step()
            opt_scheduler.step(total_loss)

            average_uncertainty_loss = 0.
            for ood_iterate in range(ood_iterations):
                ood_opt.zero_grad()
                uncertainty_loss = self.uncertainty_loss()
                uncertainty_loss.backward()
                ood_opt.step()
                average_uncertainty_loss += uncertainty_loss
            average_uncertainty_loss = average_uncertainty_loss/ood_iterations
            ood_opt_scheduler.step(average_uncertainty_loss)

            print(f"Classification Loss: {class_loss}")
            print(f"Uncertainty Loss: {average_uncertainty_loss}")

        return None