import sys
import argparse
import yaml


def get_config_modular(main_args):
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file',
                        type=str,
                        help='file to read training configs from')
    args = parser.parse_args(main_args)
    with open(args.config_file, 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return cfg


def get_config():
    return get_config_modular(sys.argv[1:])
