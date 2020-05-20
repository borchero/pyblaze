import torch
from .base import Engine
from ._utils import forward

class WGANEngine(Engine):
    """
    Engine to be used for training a Wasserstein GAN. It enables training the critic for multiple
    iterations as well as skipping the training of the generator for specific iterations.

    The GAN msut be supplied as a single model with :code:`generator` and :code:`critic` attributes.


    The engine requires data to be available in the following format:

    Parameters
    ----------
    noise: list of object
        List of n + 1 noise inputs for the generator (batch size N, noise dimension D). When
        performing predictions, there is no need to pass a list.
    real: list of object
        List of n real inputs for the critic. Shape depends on the task at hand. When performing
        predictions, this value must not be given.


    The :meth:`train` method allows for the following keyword arguments:

    Parameters
    ----------
    generator_optimizer: torch.optim.Optimizer
        The optimizer to use for the generator model.
    critic_optimizer: torch.optim.Optimizer
        The optimizer to use for the critic model.
    generator_loss: torch.nn.Module
        The loss to use for the generator. Receives a tuple of the critic's outputs for fake and
        real data :code:`(fake_out, real_out)` (both values of type `torch.Tensor [N]` for batch
        size N) and ought to return a single `torch.Tensor [1]`, the loss. Consider using
        :class:`pyblaze.nn.WassersteinLossGenerator`.
    critic_loss: torch.nn.Module
        The loss to use for the critic. Receives a 4-tuple of the critic's outputs for fake and real
        data as well as the the inputs to the critic :code:`(fake_out, real_out, fake_in, real_in)`
        (the first two values are of type `torch.Tensor [N]`, the second two values of type
        `object` depending on the data). It ought to return a tuple of
        `(torch.Tensor [1], torch.Tensor [1])` where only the first value requires a gradient, the
        actual loss. The second value ought to provide an estimation of the earth mover's distance
        (i.e. the negative loss when not penalizing gradients). Consider using
        :class:`pyblaze.nn.WassersteinLossCritic`.
    critic_iterations: int, default: 3
        The number of iterations to train the critic for every iteration the generator is trained.
        A higher value ensures that the critic is trained to optimality and provides better
        gradients to the generator. However, it slows down training.
    skip_generator: bool, default: False
        Whether to skip optimizing the generator. Usually, you would want to alter this value
        during training via some callback. This way, you can ensure that the critic is trained to
        optimality at specific milestones.
    clip_weights: tuple of (float, float), default: None
        When this value is given, all weights of the generator are clamped into the given range
        after every step of the optimizer. While this ensures Lipschitz-continuity, using different
        means of regularization should be preferred.
    """

    def train_batch(self, data,
                    generator_optimizer=None, critic_optimizer=None,
                    generator_loss=None, critic_loss=None,
                    critic_iterations=3, skip_generator=False, clip_weights=None):

        noise, real = data
        summary = {}

        # Train the generator
        if not skip_generator:
            generator_optimizer.zero_grad()

            c_fake = forward(self.model, noise[0])
            loss = generator_loss(c_fake)
            loss.backward()

            generator_optimizer.step()
            summary['loss_generator'] = loss.item()

        # Train the critic for multiple iterations
        critic_losses = []
        em_distances = []
        for i in range(critic_iterations):
            critic_optimizer.zero_grad()

            with torch.no_grad():
                fake = forward(self.model.generator, noise[i+1])

            c_fake = forward(self.model.critic, fake)
            c_real = forward(self.model.critic, real[i])

            loss, em_distance = critic_loss(c_fake, c_real, fake, real[i])
            loss.backward()

            critic_optimizer.step()

            # Clip weights if required
            if clip_weights is not None:
                with torch.no_grad():
                    for param in self.model.critic.parameters():
                        param.clamp_(*clip_weights)

            critic_losses.append(loss.item())
            em_distances.append(em_distance.item())

        summary['loss_critic'] = sum(critic_losses) / len(critic_losses)
        summary['em_distance'] = sum(em_distances) / len(em_distances)

        return summary

    def eval_batch(self, data):
        return 0

    def collate_losses(self, losses):
        losses_critic = [l['loss_critic'] for l in losses]
        losses_generator = [l['loss_generator'] for l in losses]
        em_distances = [l['em_distance'] for l in losses]
        return {
            'loss_critic': sum(losses_critic) / len(losses_critic),
            'loss_generator': sum(losses_generator) / len(losses_generator),
            'em_distance': sum(em_distances) / len(em_distances)
        }
