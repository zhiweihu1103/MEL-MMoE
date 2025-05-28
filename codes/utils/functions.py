import argparse
from omegaconf import OmegaConf


def setup_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config', type=str, default='config/wikidiverse.yaml')
    _args = parser.parse_args()
    args = OmegaConf.load(_args.config)
    return args