"""Model management utils."""
import abc
import os
import os.path
import json

import torch


class SavableModule(abc.ABC):
    """Mixin for modules to be saved via a simple interface.

    When derived, requires the superclass to be a torch Module.
    """

    @abc.abstractmethod
    def get_config(self) -> dict:
        """Return a dictionary of params for the module's constructor.

        Subclass to provide your module specific implementation.
        """

    @staticmethod
    def get_config_path(directory_path: str) -> str:
        """Return configuration path given a target directory."""
        return os.path.join(directory_path, 'config.json')

    @staticmethod
    def get_model_path(directory_path: str) -> str:
        """Return model parameters path given a target directory."""
        return os.path.join(directory_path, 'model.pt')

    def save(self, directory_path: str):
        """Save model configuration and parameters.

        Can be loaded back with :meth:`load`.
        """
        os.makedirs(directory_path, exist_ok=True)

        # Save config (extract a dedicated method?)
        with open(self.get_config_path(directory_path), 'w') as file:
            json.dump(self.get_config(), file)

        # Save parameters
        torch.save(self.state_dict(), self.get_model_path(directory_path))

    @classmethod
    def load(cls, directory_path: str) -> 'SavableModule':
        """Load model configuration and parameters.

        Previously saved via :meth:`save`.
        """
        # Load config (extract a dedicated method?)
        with open(cls.get_config_path(directory_path)) as file:
            config = json.load(file)

        network = cls(**config)
        network.load_state_dict(torch.load(cls.get_model_path(directory_path)))

        return network
