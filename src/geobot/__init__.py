import argparse
import sys

from geobot.cmd_embed import register_subcommand as register_embed
from geobot.cmd_eval_retrieval import register_subcommand as register_eval_retrieval
from geobot.cmd_index import register_subcommand as register_index
from geobot.cmd_prepare import register_subcommand as register_prepare
from geobot.cmd_qc import register_subcommand as register_qc
from geobot.cmd_train import register_subcommand as register_train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geobot",
        description="GeoBot data pipeline and baseline model tooling.",
    )
    subparsers = parser.add_subparsers(dest="command")
    register_qc(subparsers)
    register_prepare(subparsers)
    register_train(subparsers)
    register_embed(subparsers)
    register_index(subparsers)
    register_eval_retrieval(subparsers)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    exit_code = args.func(args)
    if isinstance(exit_code, int) and exit_code != 0:
        sys.exit(exit_code)
