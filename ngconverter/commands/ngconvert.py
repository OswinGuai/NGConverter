import argparse
import sys

from absl import app
from absl import flags

from ngconverter.util.configparser import load_config
from ngconverter.core.task import Task

FLAGS = None

def _parse_flags_tolerate_undef(argv):
    """Parse args, returning any unknown flags (ABSL defaults to crashing)."""
    return flags.FLAGS(sys.argv if argv is None else argv, known_only=True)


def run(main=None, argv=None):
    """Runs the program with an optional 'main' function and 'argv' list."""
    main = main or sys.modules['__main__'].main
    app.run(main=main, argv=argv, flags_parser=_parse_flags_tolerate_undef)


def main():
    global FLAGS

    parser = argparse.ArgumentParser(
        description="欢迎使用NGConverter模型转换器.")
    subparsers = parser.add_subparsers()
    create_parser = subparsers.add_parser('create')
    create_parser.add_argument(
        "configuration",
        type=str,
        help="配置文件地址.")
    create_parser.add_argument(
        "--name",
        type=str,
        help="任务名.")
    create_parser.add_argument(
        "--tensorflow",
        action="store_true",
        help="Use tensorflow model first.")

    FLAGS, unparsed = parser.parse_known_args()

    run(main=create_task, argv=[sys.argv[0]] + unparsed)

def create_task():
    config = load_config(FLAGS.configuration)
    task = Task.Builder.init_by_config(FLAGS.name, config)
    task.execute()

if __name__ == "__main__":
    main()
