from importlib import import_module


def load_command_class(name):
    """
    Given a command name and an application name, return the Command
    class instance. Allow all errors raised by the import process
    (ImportError, AttributeError) to propagate.
    """
    # TODO Update path for commands
    module = import_module('ngconverter.commands.%s' % (name))
    return module.Command()


# class CmdProxy:
#     def register(self, cmd_name, *args, **options):
#         load_command_class(cmd_name)