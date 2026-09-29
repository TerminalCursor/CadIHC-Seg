from dataclasses import dataclass, asdict
import cellpose
from cellpose import core
from numpy import linspace, logspace


def asdict_(dclass):
    """# Convert Dataclass to Dictionary

    1. Use the naive approach first
    2. Remove all underscored keys
    """
    out_dict = asdict(dclass)
    keys = list(out_dict.keys())
    for k in keys:
        if k[0] == "_":
            del out_dict[k]
    return out_dict


def RandomVariable(lo, hi):
    """# Draw Random Number

    From range [lo, hi)

    Detects if it is a float or integer for return type.
    """
    assert type(lo) == type(hi)
    if isinstance(lo, float):
        from numpy.random import rand

        return rand() * (hi - lo) + lo
    from numpy.random import randint

    return randint(lo, hi)


class ParameterInterface:
    """# Parameter Interface

    Ensure class has the `todict` functionality
    """

    def todict(self):
        return self.__dict__


@dataclass
class DatasetConfig(ParameterInterface):
    """# Dataset Configuration

    base_directory: Store the root of the dataset's directory
    """

    base_directory: str = "histology-250929/"

    def todict(self):
        return asdict_(self)


@dataclass
class ModelBuildConfig(ParameterInterface):
    """# Model Builder Configuration

    gpu: Whether or not the gpu is being used
    nchan: `3` for the CNN version of Cellpose, None for the Transformer Version
    pretrained_model: `str` if loading a pretrained model
    diam_mean: a static `30` for compatibility and consistent resizing
    """

    gpu: bool = core.use_gpu()
    diam_mean: float = 30

    @property
    def nchan(self):
        """# Get number of channels

        Returns `3` if using the CNN version of cellpose
        Returns `None` if using the Transformer version
        """
        if cellpose.version[0] == "3":
            return 3
        return None

    def todict(self):
        """# Get the config as a dictionary

        Ignore the nchan and pretrained_model if invalid/ missing
        """
        out_dict = asdict_(self)
        if self.nchan is not None:
            out_dict["nchan"] = self.nchan
        if "pretrained_model" in dir(self):
            out_dict["pretrained_model"] = self.pretrained_model
        return out_dict


@dataclass
class ModelConfig:
    """# Model Configuration

    Log if the type is the CNN or Transformer Version
    """

    base_arch: str = "cyto" if cellpose.version[0] == "3" else "SAM"
    build_config = ModelBuildConfig()


@dataclass
class AugmentConfig(ParameterInterface):
    """# Augmentation Configuration:

    Perturbation variables
    count: Number of perturbations to apply
    radius: Size of perturbation
    pinch: Strength of perturbation in region
    """

    _count_lo: int = 20
    _count_hi: int = 40

    @property
    def count(self):
        return RandomVariable(self._count_lo, self._count_hi)

    _radius_lo: int = 24
    _radius_hi: int = 48

    @property
    def radius(self):
        return RandomVariable(self._radius_lo, self._radius_hi)

    _pinch_lo: int = -18
    _pinch_hi: int = -12

    @property
    def pinch_strength(self):
        return RandomVariable(self._pinch_lo, self._pinch_hi)

    def todict(self):
        out_dict = asdict_(self)
        out_dict["count"] = self.count
        out_dict["radius"] = self.radius
        out_dict["pinch_strength"] = self.pinch_strength
        return out_dict


@dataclass
class TrainingConfig:
    """# Training Configuration

    start_/end_learning_rate: The learning rate bounds
    learn_space: Spacing of the learning rate ('linspace' or 'logspace')
    learning_rate: Current learning rate
    n_images_per_epoch: Number of images to train on before advancing cycle
    n_augments: Number of augments to add to each cycle training set
    n_gen: Number of generated samples to add to each cycle training set
    optimizer: Optimizer name ("ADAM" or "SGD")
    weight_decay: Weight decay to use
    epochs: Number of epochs to train
    normalize: Whether to have cellpose normalize the data
        (Recommended False) as our augmentations may include data we shouldn't normalize
    cycle: Current training cycle
    training_cycles: Number of total training cycles to perform
    """

    start_learning_rate: float = 3e-4
    end_learning_rate: float = 1e-5
    learn_space: str = "linspace"
    learning_rate: float = 0.0
    n_images_per_epoch: int = 64
    n_augments: int = 8
    n_gen: int = 0
    optimizer: str = "ADAM"
    weight_decay: float = 0.0
    epochs: int = 1000
    normalize: bool = False
    cycle: int = 0
    training_cycles: int = 1000

    def __post_init__(self):
        """# Ensure to set the learning rate properly"""
        self.setup_cycles()

    def setup_cycles(self, n_cycles=None):
        """# Set the Number of Cycles to Train On

        Additionally, set the learning rate progression for the cycle count
        """
        if n_cycles is not None:
            self.training_cycles = n_cycles
        from numpy import log10

        assert self.learn_space in ["linspace", "logspace"]
        l, h = self.start_learning_rate, self.end_learning_rate
        if self.learn_space[:3] == "log":
            l = log10(l)
            h = log10(h)
        space = eval(self.learn_space)
        self._learning_rates = space(l, h, self.training_cycles)
        self.learning_rate = self._learning_rates[self.cycle]

    def next_cycle(self):
        """# Next Cycle Logic

        Advance Learning Rate Schedule
        """
        self.cycle += 1
        if self.cycle < len(self._learning_rates):
            self.learning_rate = self._learning_rates[self.cycle]

    def cycle_kwargs(self):
        """# Cellpose Cycle Keyword Args

        Build the cellpose training cycle kwargs
        """
        kwargs = {
            "normalize": self.normalize,
            "n_epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "SGD": self.optimizer.lower() == "sgd",
            "nimg_per_epoch": self.n_images_per_epoch,
        }
        return kwargs


@dataclass
class UnifiedConfig:
    """# Unified Config

    Dataset Config, Model Config, Augment Config, and Training Config

    pretrained(<str>): For loading pretrained models
    tostr(): Reporting the config state in a quick manner
    """

    dataset = DatasetConfig()
    model = ModelConfig()
    augments = AugmentConfig()
    training = TrainingConfig()

    def pretrained(self, pretrained_model):
        """# Set the pretrained model"""
        self.model.build_config.pretrained_model = pretrained_model

    def tostr(self):
        """# Report Config State"""
        from json import dumps

        dump = lambda x: dumps(x, indent=4)
        dstr = dump(self.dataset.todict())
        mstr = dump(
            {"base": self.model.base_arch, "kwargs": self.model.build_config.todict()}
        )
        astr = dump(self.augments.todict())
        tstr = dump(asdict_(self.training))
        return f"{dstr}\n{mstr}\n{astr}\n{tstr}\n"
