"""Learned policies, run through LeRobot. Needs the `policy` extra.

An *untrained* ACT is the point for now: random weights still exercise every
stage a trained one will -- image and state preprocessing, normalisation,
action chunking, de-normalisation, and the env's clipping -- so the whole
pipe is proven before there is a dataset to train on. Swapping in a trained
checkpoint changes where the weights and statistics come from, not this code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from learning.env import PickCubeEnv

#: ImageNet statistics: what a ResNet backbone expects its input normalised
#: by, trained or not.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _placeholder_stats(env: PickCubeEnv) -> dict:
    """Normalisation statistics without a dataset.

    State and action are centred on the middle of their ranges with a quarter
    of the range as one standard deviation, which keeps an untrained network's
    outputs near home rather than at a joint limit. A trained policy replaces
    these with its dataset's own.
    """
    import torch

    low = env.action_space.low.astype(np.float32)
    high = env.action_space.high.astype(np.float32)
    span = {
        "mean": torch.from_numpy((low + high) / 2),
        "std": torch.from_numpy((high - low) / 4),
    }
    stats = {"observation.state": dict(span), "action": dict(span)}
    for camera in env.cameras:
        stats[f"observation.images.{camera.name}"] = {
            "mean": torch.tensor(IMAGENET_MEAN).view(3, 1, 1),
            "std": torch.tensor(IMAGENET_STD).view(3, 1, 1),
        }
    return stats


class ActPolicy:
    """LeRobot's ACT, sized to this env's cameras and 7-D state and action."""

    def __init__(self, env: PickCubeEnv, chunk_size: int = 30, seed: int = 0):
        import torch
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        torch.manual_seed(seed)
        state_dim = env.action_space.shape[0]
        inputs = {"observation.state": PolicyFeature(FeatureType.STATE, (state_dim,))}
        for camera in env.cameras:
            inputs[f"observation.images.{camera.name}"] = PolicyFeature(
                FeatureType.VISUAL, (3, camera.height, camera.width)
            )
        config = ACTConfig(
            input_features=inputs,
            output_features={"action": PolicyFeature(FeatureType.ACTION, (state_dim,))},
            chunk_size=chunk_size,
            n_action_steps=chunk_size,
            # Untrained end to end, so no ImageNet download either.
            pretrained_backbone_weights=None,
            device="cpu",
        )
        self.policy = ACTPolicy(config)
        self.policy.eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            config, dataset_stats=_placeholder_stats(env)
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "ActPolicy":
        """A trained ACT, from a `lerobot-train` checkpoint directory.

        `path` is the `pretrained_model` folder, e.g.
        outputs/act_pick_cube/checkpoints/last/pretrained_model. Trained on a
        GPU, run here on the CPU: both the weights and the preprocessor's
        device step are moved, or the first batch is sent to a missing CUDA.
        """
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        config = ACTConfig.from_pretrained(path)
        config.device = "cpu"
        self = cls.__new__(cls)
        self.policy = ACTPolicy.from_pretrained(path, config=config)
        self.policy.eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            config,
            pretrained_path=str(path),
            preprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        return self

    def reset(self):
        self.policy.reset()

    def select_action(self, observation) -> np.ndarray:
        import torch
        from lerobot.envs.utils import preprocess_observation

        with torch.inference_mode():
            batch = self.preprocess(preprocess_observation(observation))
            action = self.postprocess(self.policy.select_action(batch))
        return action.squeeze(0).numpy()

    def planned_actions(self) -> list[np.ndarray]:
        """The rest of the current chunk, de-normalised into joint targets.

        ACT queues its chunk in normalised units, so the queue has to go
        through the same postprocessor the executed actions do.
        """
        import torch

        queue = list(self.policy._action_queue)
        if not queue:
            return []
        with torch.inference_mode():
            actions = self.postprocess(torch.cat(queue, dim=0))
        return list(actions.numpy())


class RandomPolicy:
    """Uniform samples from the action space -- the floor any policy must beat."""

    def __init__(self, env: PickCubeEnv, seed: int = 0):
        self.space = env.action_space
        self.space.seed(seed)

    def reset(self):
        pass

    def select_action(self, observation) -> np.ndarray:
        return self.space.sample()
