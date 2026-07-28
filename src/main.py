import hydra
from omegaconf import DictConfig

from src.datasets import *
from src.registry import DATASETS


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="config"
)
def main(cfg: DictConfig):

    dataset = DATASETS.build(
        cfg.dataset.name,
        root_dir=cfg.dataset.root_dir,
    )

    print(dataset)


if __name__ == "__main__":
    main()