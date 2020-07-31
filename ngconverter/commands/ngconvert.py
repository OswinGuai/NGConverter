import argparse
import os
import logging

logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from ngconverter.util.configparser import load_config
from ngconverter.record.redirect_stdout import close_redirection, direct_to_file, direct_to_console


parser = argparse.ArgumentParser(description="欢迎使用NGConverter模型转换器.")
subparsers = parser.add_subparsers(title='操作', help='帮助', dest='操作选项{create}', required = True)
create_parser = subparsers.add_parser('create')
create_parser.add_argument(
        "--name",
        required=True,
        type=str,
        help="任务名.")
create_parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="配置文件地址.")
create_parser.add_argument(
        "--tensorflow",
        action="store_true",
        help="优先使用TensorFlow.")

args = parser.parse_args()

def main():
    create_task()

def create_task():
    config = load_config(args.config)
    from ngconverter.core.task import Task
    task = Task.Builder.init_by_config(args.name, config)

    direct_to_console()
    # Do task
    task.execute()
    # close redirection. Suppose there will be a redirection during the task.
    close_redirection()

if __name__ == "__main__":
    main()
