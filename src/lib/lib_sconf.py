"""
General purpose preprocessing library
=====================================

.. module:: lib_sconf
   :platform: Linux
   :synopsis: tool library to manage argument parsing, logging, path configuration, object saving, etc.

.. moduleauthor:: Jose M. Esnaola-Acebes <josemesnaola@gmail.com>


Parsing methods
---------------

.. autosummary::
   :toctree: generated/

   parser_init       Initializes parsing of arguments from an external configuration file.
   parser            Completes the initial parsing taking command-line arguments.

Logging and debugging methods and objects
-----------------------------------------

.. autosummary::
   :toctree: generated/

   log_conf          Sets up logging configuration.
   MyLog             A class that makes use of :func:`msg`.
   debug_func        Decorator that prints the function signature ant its return value.
   count_calls       Decorator that counts the number of calls of the decorated function.

Path-related methods
--------------------

.. autosummary::
   :toctree: generated/

   get_paths         Gets the absolute and relative paths of a given file.
   create_dir        Creates a directory if it does not exist.
   path_exists       Checks whether a path exists or not.
   check_overwrite   Checks whether a path exists and overwrites it if necessary.

Saving and loading methods
--------------------------

.. autosummary::
   :toctree: generated/

   save_obj          Save a given object at a given path.
   load_obj          Loads an object from a given path.

Miscellaneous objects
---------------------

.. autosummary::
   :toctree: generated/

   now               Gets a string with the date and daytime.
   yesno_question    Asks a target question with Yes or No answer.
   isnotebook        Checks whether the execution is being held in a notebook.
   print_at          Prints a string at a specific location of the terminal's screen.
   LimitedSizeDict   Creates a dictionary of limited size.
   dict_depth        Computes the depth of a dictionary.
   fmt_latex_table   Formats a given table (dictionary) into a latex table
   truncate          Truncates a floating point number up to a given decimal

Implementation
--------------

.. todo::

   Implementation notes.
"""

import argparse

try:
    import yaml
except ImportError:
    raise ImportError(
        'Install pyyaml module: pip install PyYAML\tor\tconda install pyyaml')
import os
import sys
import pathlib
from typing import Union
import datetime
import logging.config
import pickle
from collections import OrderedDict
from collections.abc import Iterable
import functools

try:
    import readchar
except ImportError:
    raise ImportError("Install `readchar` module: pip install readchar")

try:
    from colorlog import ColoredFormatter
except ImportError:
    raise ImportError("Install `colorlog` module: pip install colorlog")

__author__ = 'Jose M. Esnaola-Acebes'
__docformat__ = 'reStructuredText'

# logging.getLogger('lib_sconf').addHandler(logging.NullHandler())
log = None
ESC = "\x1b"
(RED, GREEN, YELLOW, CYAN, WHITE) = [ESC + "[%dm" % c for c in [31, 32, 33, 36, 37]]
(BRED, BGREEN, BYELLOW, BCYAN, BWHITE) = [ESC + "[1;%dm" % c for c in [31, 32, 33, 36, 37]]
(UP, DOWN, RIGHT, LEFT) = [ESC + "[1%s" % letter for letter in ['A', 'B', 'C', 'D']]
DEL_LINE = ESC + "[2k"

EXAMPLE_CONF = """
Block example title:
  -a --argument:
    description: "Example of how to introduce list arguments."
    default:     [1]
    name:        "<argument>"
    choices:     ~
  -anthr --another:
    description: "Another example, for float arguments."
    default:     0.0
    name:        "<another>"
    choices:     ~
Another Block example title:
  -s --string:
    description: "String argument example with choices."
    default:     'string'
    name:        "<string>"
    choices:     ['string1', 'string2']
  -b --boolean:
    description: "Boolean argument example with choices."
    default:     False
    name:        "<bool>"
    choices:     [False, True]
"""


class Parser(argparse.ArgumentParser):
    """Wrapper class that helps parsing command-line arguments whose description is based on a configuration
    file or a configuration string or a configuration dictionary. The configuration file or string must have
    PyYaml format.

    Examples
    ^^^^^^^^

     - Use a rich format configuration file or string:
    """

    def __init__(self, args=None, desc=None, usage=None, groups=(None,), conf=None, debug=False, **kwargs):
        """Initialization of the class: if ``conf`` is of type :py:class:`str`, first tries to load a
        possible configuration file ``conf``. If this fails, tries to interpret the configuration as
        a PyYaml string.

        If ``conf`` is a dictionary, use this dictionary to initialize the parser. If no ``conf``
        is given, try a initial parsing with a basic parsing configuration and look for a configuration file
        passed through command line arguments.

        :param list of str or None args: list of arguments that will be parsed.
        :param str desc: Description to be printed in terminal.
        :param str usage: Help description to be printed whith ``-h`` or ``--help`` cmd arguments.
        :param list of str groups or tuple of str groups: list of groups used to classify the arguments.
        :param str or None or dict conf: configuration file or string, or dictionary.
        :param bool debug: whether to debug the parsing process.
        :param kwargs: additional keyword arguments passed to :func:`__call__`
        """
        if args is None:
            args = sys.argv[1:]
        logging.debug('Initiating logging module.')
        if args:
            logging.root.setLevel(10 if debug or ('DEBUG' in args) else 20)
        self._log = logging.getLogger('lib_sconf.Parser')
        self._log.debug('Initializing Parser.')

        # Initialization of the Parser
        usage = f"python {sys.argv[0]:s} [-O <options>]" if usage is None else usage
        argparse.ArgumentParser.__init__(self, add_help=False, description=desc, usage=usage)
        # Variables of this class
        self._args = args
        self._gr_labels = groups
        self._first_args = None
        self._second_args = None
        self.conf_path = None
        self.debug_level = None
        self.help = False
        self._help_command = False
        self._conf = conf
        self._conf_loaded = False
        self._req_args = None
        self._n_opt = 0
        self._n_req = 0
        # Dictionary and namespace
        self.opts = {}
        self.args = None
        self.groups = {}

        # Set up the parser
        self.__call__(args, conf, **kwargs)

        # Check if the debug level is correctly set
        if self.debug_level is None:
            self.debug_level = vars(self.args)['db']
            self._log.setLevel(logging.getLevelName(self.debug_level))

    def __call__(self, args=None, conf=None, required_args=None, add_groups=False, force_conf=False, parse=True):
        """Search for a valid configuration, set up the parsing configuration, and parse.

        :param list of str or None args: list of arguments that will be parsed.
        :param dict or str or None conf: configuration file or string or dictionary.
        :param dict or None required_args: required arguments dictionary.
        :param bool add_groups: whether to add grouped arguments to the main dictionary ``opts``.
        :param bool force_conf: whether to force the configuration.
        :param bool parse: whether to parse arguments now.
        """
        if args is not None:
            self._args = args
        if required_args is not None:
            self._req_args = required_args

        if not self._conf_loaded or force_conf:
            if conf is not None:
                self._conf = conf
            # Try to load a configuration using ``conf``
            if not isinstance(self._conf, dict):  # No configuration has been loaded yet.
                self._conf = self._load_conf_file(conf)
            if not isinstance(self._conf, dict):  # No configuration has been loaded yet.
                self._initial_parsing()  # try to load it using command line arguments.

            try:
                assert isinstance(self._conf, dict)
            except AssertionError:
                self._log.error('Parser configuration could not be loaded.')
                return

            # Configure the parsing
            self._log.debug('Configuring the parser.')
            self._load_conf(self._req_args)
            self._conf_loaded = True
        # Parse the arguments
        if parse and self._conf_loaded:
            self._parse_args(self._args, add_groups)

    def _load_conf_file(self, conf):
        """PyYaml style configuration loader.

        :param str or None conf: configuration file or string.
        """
        _conf = None
        if isinstance(self._conf, str):
            # Try to load a configuration from a file
            self._log.debug(f"Trying to read a configuration file ({conf}).")
            try:
                with open(conf, 'rb') as f:
                    _conf = yaml.load(f, Loader=yaml.FullLoader)
            except IOError:
                self._log.debug("Configuration file not found or cannot be read.")
                self._log.debug("Falling back to PyYaml loading of the string.")
                _conf = yaml.load(conf, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                self._log.error("Error loading configuration, bad format.", exc)
        return _conf

    def _initial_parsing(self):
        """Method that tries to find a configuration file in the command-line arguments that will
        be used to parse the rest of the commands.
        """
        # Add first order arguments to configuration
        self.add_argument('-f', '--file', default="conf.txt", dest='-f', metavar='<file>')
        self.add_argument('-db', '--debug', default="INFO", dest='db', metavar='<debug>',
                          choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
        # Check for arguments matching the latter options
        self._first_args, self._second_args = self.parse_known_args(self._args)
        self.conf_path = vars(self._first_args)['-f']
        self.debug_level = vars(self._first_args)['db']
        self._log.setLevel(logging.getLevelName(self.debug_level))
        # Search for help option among the non-parsed arguments
        self.help = True if set(self._second_args).intersection({'-h', '--help'}) else False
        # Try to load the configuration file if any
        try:
            with open(self.conf_path, 'rb') as f:
                self._conf = yaml.load(f, Loader=yaml.FullLoader)
        except IOError:
            self._log.debug("Configuration file not found or cannot be read.")
        except yaml.YAMLError as exc:
            self._log.error("Error loading configuration, bad format.", exc)

    def _load_conf(self, req_args=None):
        """Set up the parsing configuration using a configuration dictionary.

        The configuration may contain groups, such that arguments are grouped using those groups.

        :param dict or None req_args: required arguments dictionary.
        """
        self._n_opt = 0
        depth_of_conf = dict_depth(self._conf)
        # Load configuration of optional arguments
        if depth_of_conf > 1:  # If the configuration dictionary has a rich format (name, description, etc.)
            if depth_of_conf == 2:
                self._conf = {"Options": self._conf}
            for group in self._conf:  # Search and store arguments by groups
                if group in self._gr_labels:
                    self.groups[group] = {}
                gr = self.add_argument_group(group)  # Group
                ops = self._conf[group]  # Set of arguments inside group
                for key in self._conf[group]:  # Search keyword arguments in each group
                    if key in ('h', 'help', '-h', '--help'):
                        self._log.warning(f"Key {key} is normally reserved for the help command, overwriting.")
                    self._n_opt += 1
                    flags = key.split()
                    def_v = ops[key].get("default", "null")
                    descr = ops[key].get("description", "No description.")
                    mname = ops[key].get("name", f"meta{self._n_opt:02d}")
                    chois = ops[key].get("choices", [0, 1])
                    v_nam = flags[0][1:]

                    try:
                        if isinstance(def_v, bool):  # Distinguish between bool, list, and others.
                            gr.add_argument(*flags, default=def_v, help=descr, dest=v_nam, action='store_true')
                        elif getattr(def_v, '__getitem__', False) and not isinstance(def_v, str):
                            tipado = type(def_v[0])
                            gr.add_argument(*flags, default=def_v, help=descr, dest=v_nam, metavar=mname, type=tipado,
                                            choices=chois, nargs='+')
                        else:
                            gr.add_argument(*flags, default=def_v, help=descr, dest=v_nam, metavar=mname,
                                            type=type(def_v), choices=chois)
                        if group in self._gr_labels:
                            self.groups[group][flags[0][1:]] = def_v
                    except argparse.ArgumentError:
                        self._log.debug(f'Conflicting option string: {flags}. Skipping.')
                        self._n_opt -= 1
        else:  # If the configuration dictionary if of type: {name: value, -name: value}
            for key, value in self._conf.items():
                if key in ('h', 'help', '-h', '--help'):
                    self._log.warning(f"Key {key} is normally reserved for the help command, overwriting.")
                self._n_opt += 1
                if not isinstance(key, str):
                    key = f"{key}"
                flags = (key, "-" + key) if key.startswith('-') else ("-" + key, "--" + key)
                def_v = value
                v_nam = flags[0][1:]
                if isinstance(def_v, bool):
                    self.add_argument(*flags, default=def_v, dest=v_nam, action='store_true')
                elif getattr(def_v, '__getitem__', False) and not isinstance(def_v, str):
                    tipado = type(value[0])
                    self.add_argument(*flags, default=def_v, dest=v_nam, type=tipado, nargs='+')
                else:
                    self.add_argument(*flags, default=def_v, dest=v_nam, type=type(def_v))
        # Manually try to add help command
        try:
            self.add_argument('-h', '--help', help='Print help', default=False, action='store_true')
            self._help_command = True
            self._n_opt += 1
        except argparse.ArgumentError:
            self._log.debug(f"Conflicting option string: '-h --help'. Skipping.")
        # Load configuration of required arguments (if any)
        if req_args:
            self._log.debug(f"Configuring required arguments.")
            for arg, options in req_args.items():
                try:
                    self.add_argument(arg, **options)
                    self._n_req += 1
                except argparse.ArgumentError:
                    self._log.debug(f"Conflicting option string: '-h --help'. Skipping.")

    def _parse_args(self, args, add_groups=False):
        """Method that parses the arguments once a configuration has been loaded.

        :param list of str args: list of arguments that will be parsed.
        :param bool add_groups: whether to add grouped arguments to the main dictionary ``opts``.
        """
        # Parse the arguments
        self._log.debug('Parsing arguments...')
        if self._help_command and set(args).intersection({'-h', '--help'}):
            self.print_help()
            self.exit(0)
        self.args = self.parse_args(args)
        self.opts = vars(self.args)
        # Modify group dictionaries
        for group in self.groups:
            for key in self.groups[group]:
                self.groups[group][key] = self.opts[key]
            if add_groups:
                self.opts[group.lower()] = self.groups[group]

    def print_conf(self, conf=None):
        """Prints the configuration file or string passed as argument ``conf`` or print the already loaded
        configuration. If ``conf='example'``, print an example configuration.

        :param str or None conf: configuration file or string.
        """
        if conf is None:
            conf = self._conf
        elif conf != 'example':
            conf = self._load_conf_file(conf)
        if conf is None or conf == 'example':  # No configuration is loaded, and ``conf`` is invalid. Print example.
            print(EXAMPLE_CONF)
        else:
            print(conf)


class Options:
    def __init__(self):
        pass


def parser_init(arguments=None):
    """ Function to handle arguments from CLI:
        First Parsing -  We parse optional configuration files.
    """

    if arguments is None:
        arguments = sys.argv[1:]
    pars = argparse.ArgumentParser(add_help=False)
    pars.add_argument('-f', '--file', default="conf.txt", dest='-f', metavar='<file>')
    pars.add_argument('-db', '--debug', default="INFO", dest='db', metavar='<debug>',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    # Check for arguments matching the latter options
    args = pars.parse_known_args(arguments)
    conf_file = vars(args[0])['-f']  # Configuration file (if any)
    debug_level = vars(args[0])['db']
    hlp = False
    for k, op in enumerate(args[1]):
        if op == '-h' or op == '--help':
            hlp = True
    return conf_file, debug_level, args, hlp


def parser(config_file, arguments=None, description=None, usage=None, groups=('Parameters',)):
    """ Function to handle arguments from CLI:
        Second Parsing -  We parse simulation parameters.

    :param str config_file: YAML format. See config_doc variable and YAML documentation for more help.
    :param list arguments: if there is a previous parsing, introduce the arguments here.
    :param str description: Description of the script/program (string)
    :param str usage: How to use the script. (string)
    :param groups: Special groups to take into account
    :type groups: tuple(str)
    """

    global log
    try:
        log.debug('Starting second parsing.')
    except (NameError, AttributeError):
        log = logging.getLogger('sconf')
    config_doc = """
        Block example title:
          -a --argument:
            description: "Example of how to introduce list arguments."
            default:     [1]
            name:        "<argument>"
            choices:     ~
          -anthr --another:
            description: "Another example, for float arguments."
            default:     0.0
            name:        "<another>"
            choices:     ~
        Another Block example title:
          -s --string:
            description: "String argument example with choices."
            default:     'string'
            name:        "<string>"
            choices:     ['string1', 'string2']
          -b --boolean:
            description: "Boolean argument example with choices."
            default:     False
            name:        "<bool>"
            choices:     [False, True]
    """

    # Opening the configuration file to load parameters
    options = None
    gprmts = {}
    ops = Options()
    try:
        if config_file.startswith(r"#conf"):
            options = yaml.load(config_file, Loader=yaml.FullLoader)
        else:
            options = yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader)
    except IOError:
        log.error("The configuration file '%s' is missing" % config_file)
        log.info("No configuration loaded.")
        options = yaml.load(config_doc, Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
        log.error("Error in configuration file:", exc)
        exit(-1)

    # Loading parameters from the 'options' dictionary and CLI options
    if usage is None:
        usage = 'python %s [-O <options>]' % sys.argv[0]
    pars = argparse.ArgumentParser(
        description=description,
        usage=usage)
    for group in options:  # Search and store arguments by groups
        if group in groups:
            gprmts[group] = {'dummy': 0}
        gr = pars.add_argument_group(group)
        args = options[group]
        for key in options[group]:  # Search keyword arguments in each group
            flags = key.split()
            if group in groups:
                gprmts[group][flags[0][1:]] = args[key]['default']
            if isinstance(args[key]['default'], bool):  # Distinguish between bool, list, and others.
                gr.add_argument(*flags, default=args[key]['default'], help=args[key]['description'], dest=flags[0][1:],
                                action='store_true')
            elif isinstance(args[key]['default'], list):
                tipado = type(args[key]['default'][0])
                gr.add_argument(*flags, default=args[key]['default'], help=args[key]['description'], dest=flags[0][1:],
                                metavar=args[key]['name'], type=tipado,
                                choices=args[key]['choices'], nargs='+')
            else:
                gr.add_argument(*flags, default=args[key]['default'], help=args[key]['description'], dest=flags[0][1:],
                                metavar=args[key]['name'], type=type(args[key]['default']),
                                choices=args[key]['choices'])

    # We parse command line arguments:
    if arguments:
        arg = arguments[1]
        opts = vars(pars.parse_args(arg))
        args = pars.parse_args(arg, namespace=ops)
    else:
        opts = vars(pars.parse_args())
        args = pars.parse_args(namespace=ops)

    # We create separate dictionaries for "special groups" (see function arguments)
    for groupkey in gprmts.keys():
        gprmts[groupkey].pop("dummy")
        for key in gprmts[groupkey].keys():
            gprmts[groupkey][key] = opts[key]
        opts[groupkey.lower()] = gprmts[groupkey]

    return opts, args


def log_conf(db, config_file=None, name='simulation', logdir='./log'):
    """ Logging configuration. Sets the logging level and the format of display using a
    configuration file ``conf_file`` if provided.

    :param str db: debugging level, must be an attribute in logging. See help(logging).
    :param str config_file: external logging configuration file for handlers configuration.
    :param str name: name of the logger.
    :param str logdir: Directory where the log file is stored.
    """
    global log
    logging_doc = """
        version: 1
        formatters:
          simple:
            format: "[%(levelname)-8.8s]:%(name)-20.20s:%(funcName)-10.10s:\n\t%(message)s"
        handlers:
          console:
            class: logging.StreamHandler
            level: DEBUG
            formatter: simple
            stream: ext://sys.stdout
          file:
            class: logging.FileHandler
            level: DEBUG
            formatter: simple
            filename: 'log/""" + name + """.log'
        loggers:
          simulation:
            level: DEBUG
            handlers: [console, file]
            propagate: no
        root:
          level: DEBUG
          handlers: [console, file]
    """

    # Setting debug level
    debug = getattr(logging, db.upper(), None)
    # Check the value
    if not isinstance(debug, int):
        raise ValueError('Invalid log level: %s' % db)
    # Check logging folder (default is log)
    cwd = os.getcwd()
    if not os.path.exists(logdir):
        try:
            os.mkdir(logdir)
        except IOError:
            raise IOError('Path %s/%s does not exist.' % (cwd, logdir))
    filename = "%s/%s.log" % (logdir, name)
    if os.path.exists(filename):
        f = open(filename, 'a+')
    else:
        f = open(filename, 'w+')
    day, hour = now()
    f.write("\n[%s\t%s]\n" % (day, hour))
    f.write("-------------------------\n")
    f.close()

    # Output format
    logformat = "%(log_color)s[%(levelname)-8.8s]%(reset)s %(name)-12.12s:%(funcName)-8.8s: " \
                "%(log_color)s%(message)s%(reset)s"
    formatter = ColoredFormatter(logformat, log_colors={
        'DEBUG': 'cyan',
        'INFO': 'white',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    })

    # General Configuration
    if config_file:
        logging.config.dictConfig(yaml.load(open(config_file, 'r'), Loader=yaml.FullLoader))
    else:
        logging.config.dictConfig(yaml.load(logging_doc, Loader=yaml.FullLoader))

    # Handler
    handler = logging.root.handlers[0]
    handler.setLevel(debug)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    log = logging.getLogger('lib_sconf')
    log.debug("Logger succesfully set up. Starting debuging, level: %s" % db)

    mylog = MyLog(debug)
    return logger, mylog


class MyLog:
    """A class that handles the deployment of :func:`MyLog.msg` taking into account the
    debugging level.
    """

    def __init__(self, level):
        """

        :param int level: debugging level, following :py:class:`logging` similar scale.
        """
        self._level = level
        self._levels = dict(debug=10, info=20, warning=30, error=50, critical=100)

    def __call__(self, passed=0, level='info', del_pb=False):
        """Log an OK/FAILED message on the previously (line above) logged/printed message.

        :param int passed: code to indicate the type of message. (0: OK, -1: FAILED warn, any: FAILED error)
        :param str level: debugging level ('debug', 'info', 'error', 'critical')
        :param bool del_pb: whether to delete the current line's content.
        """
        if self._level <= self._levels[level]:
            self.msg(passed, del_pb)

    # noinspection PyTypeChecker
    @staticmethod
    def msg(error, del_pb=False, flush=True, up_lines=1):
        """Prints (or deletes) a colored OK/FAILED in a previously printed line.

        :param int error: integer value that determines the what to print.
        :param bool del_pb: whether to delete the current line before printing.
        :param bool flush: whether to flush the stdout buffer or not (default is True).
        :param int up_lines: number of lines to move the cursor upwards.
        """
        # Save current cursor position
        print(ESC + "[s", end='', flush=True)  # To get the cursor is necessary to flush.
        if del_pb:
            print(DEL_LINE, end='', flush=flush)
        if error == 0:
            print(UP * up_lines + RIGHT + GREEN + "OK".ljust(8) + WHITE, flush=flush)
        elif error == -1:
            print(UP * up_lines + RIGHT + YELLOW + "FAILED".ljust(8) + WHITE, flush=flush)
        else:
            print(UP * up_lines + RIGHT + RED + "FAILED".ljust(8) + WHITE, flush=flush)
        # Go back to the initial cursor position
        print(ESC + "[u", end='', flush=flush)


def debug_func(func):
    """Decorator: Print the function signature and return value"""

    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]  # 1
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # 2
        signature = ", ".join(args_repr + kwargs_repr)  # 3
        logging.debug(f"Calling {func.__name__}({signature})")
        value = func(*args, **kwargs)
        logging.debug(f"{func.__name__!r} returned {value!r}")  # 4
        return value

    return wrapper_debug


def count_calls(func):
    """Decorator: count number of calls of func."""

    @functools.wraps(func)
    def wrapper_count_calls(*args, **kwargs):
        wrapper_count_calls.num_calls += 1
        print(f"Call {wrapper_count_calls.num_calls} of {func.__name__!r}")
        return func(*args, **kwargs)

    wrapper_count_calls.num_calls = 0
    return wrapper_count_calls


def now(daysep=', ', hoursep=':'):
    """ Returns datetime

    :param str daysep: separation used for the date.
    :param str hoursep: separation used for the hour.
    :return: day and hour tuple.
    :rtype: Tuple[str, str]
    """
    _now = datetime.datetime.now().timetuple()[0:6]
    sday = daysep.join(map(str, _now[0:3]))
    shour = hoursep.join(map(str, _now[3:]))
    return sday, shour


def get_paths(filename: str) -> dict:
    """
    Gets the real path of the ``filename``, its directory and the current working directory.

    :param filename: Name of file (it can be a relative path, an absolute path or just the file name).
    :return: dictionary including the file path (``s_path``), directory (``s_dir``) and
             current working directory (``dir``).

    """

    scriptpath = os.path.realpath(filename)
    scriptdir = os.path.dirname(scriptpath)
    cwd = os.getcwd()
    return {'s_path': scriptpath, 's_dir': scriptdir, 'dir': cwd}


def create_dir(folder_path: str) -> int:
    """
    Create a directory at ``folder_path`` if it does not exist yet.

    :param folder_path: name of the desired folder. It can be a relative path or an absolute path.
    :return: 0 if no exceptions were raised.
    """

    if not path_exists(folder_path):
        try:
            os.makedirs(folder_path)
        except OSError:
            raise IOError('Directory %s exists or could not be created.' % folder_path)
    return 0


def path_exists(path):
    return os.path.exists(path)


def check_overwrite(filename: Union[str, pathlib.Path], **kwargs) -> pathlib.Path:
    """
    Checks whether the path of the target filename exists. If it does, asks to overwrite or change name of file.

    :param filename: name or path of the file to be created or overwritten.
    :keyword bool force: whether to force overwriting (if necessary) without asking, *defaults to False*
    :keyword bool auto: whether to automatically create a new file with an appended string, *defaults to True*
    :keyword Union[None, str, bytes] append: Optional appendix to be added to the filename in case it already exists.
    :keyword int lzeros: number of digits that the version will have with leading zeros, *defaults to 3*
    :return: path of ``filename`` or of the new file name.
    """
    filename = pathlib.Path(filename)
    parent = filename.parent
    suffix = filename.suffix
    name = filename.name.removesuffix(suffix)
    file_v = 1
    append, append_fmt = kwargs.get('append', None), f"_%0{kwargs.get('lzeros', 3)}d"
    while filename.exists():
        logging.warning(f"File '{filename}' already exists.")
        if kwargs.get('force', False):
            break
        elif kwargs.get('auto', True):
            if append is None:
                filename = parent.joinpath(name + append_fmt % file_v + suffix)
                file_v += 1
            else:
                filename = parent.joinpath(name + append + suffix)
                append_fmt = append + append_fmt
                append = None
        else:
            if yesno_question('Do you want to overwrite %s?' % filename):
                break
            else:
                filename = parent.joinpath(input('Write a new name for the file: '))

    return filename


def yesno_question(message, default_answer='N'):
    """Simple terminal oriented yes/no question generator.

    :param str message: question to show in prompt.
    :param str default_answer: default answer (When ENTER is hit).
    :return: whether the answer was yes or no.
    :rtype: bool
    """
    options, response = ('(y/N)', False) if default_answer.capitalize().startswith('N') else ('(Y/n)', True)
    yes_no = ''
    while not yes_no.capitalize().startswith(('Y', 'N')) and yes_no != '\r':
        print(message + options, end='', flush=True)
        yes_no = readchar.readchar()
        print(yes_no)
    if yes_no.capitalize().startswith('Y'):
        response = True
    elif yes_no.capitalize().startswith('N'):
        response = False

    return response


def save_obj(obj, name, directory='.', extension='.pkl'):
    """Save ``obj`` at path given by ``name`` or ``directory/name``.

    :param object obj: object to be saved in binary format.
    :param str name: name of the file.
    :param str directory: name of the directory where the ``obj`` will be saved.
    :param str extension: extension used to save the object.
    :return: True if the process was successful, False if not.
    :rtype: bool
    """
    path, filename = os.path.split(name)
    if path == '':
        path = directory
    path = os.path.join(path, filename)

    try:
        with open(path + extension, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    except IOError:
        logging.error(f'Saving object at {path + extension} failed.')
        return False
    return True


def load_obj(name, directory='.', extension='.pkl'):
    """Load an ``object`` from ``name`` or ``directory/name``.

    :param str name: name of the file.
    :param str directory: name of the directory where the file is.
    :param str extension: extension of the loaded file-name.
    :return: loaded object or False
    """
    path, filename = os.path.split(name)
    if path == '':
        path = directory
    path = os.path.join(path, filename)

    try:
        with open(path + extension, 'rb') as f:
            return pickle.load(f)
    except IOError:
        logging.error(f'Loading object from {path + extension} failed.')
        return False


def isnotebook():
    """Check whether the execution is being done inside a notebook (``ipython`` or ``jupyter``)."""
    try:
        # noinspection PyUnresolvedReferences
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


class LimitedSizeDict(OrderedDict):
    """Modified ordered-dictionary which has its size limited by ``size_limit``."""

    def __init__(self, *args, **kwargs):
        """See :py:class:`dict` for the general documentation on dictionaries and
        :py:class:`collections.OrderedDict` for documentation about ordered dictionaries.
        This custom implementation sets a limit to the size of the dictionary by popping the oldest
        stored item when the ``size_limit`` has been reached.
        """

        self.size_limit = kwargs.pop("size_limit", None)
        OrderedDict.__init__(self, *args, **kwargs)
        self._check_size_limit()

    def __setitem__(self, key, value, **kwargs):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        """Check if size limit has been reached, and pop the oldest element of the dictionary in that case."""
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)


def print_at(text, x_cursor=0, y_cursor=0, flush=True, cursor=True):
    """Print ``text`` at the specified coordinates through stdout. Origin (0, 0) is the current
    cursor position. It uses ANSI escape (ESC) commands to move through the terminal's screen.
    See for example <https://es.wikipedia.org/wiki/C%C3%B3digo_escape_ANSI>`_ for additional details
    about these commands.

    :param str text: text to be printed.
    :param int x_cursor: horizontal coordinate.
    :param int y_cursor: vertical coordinate.
    :param bool flush: whether to flush the stdout buffer or not (default: True).
    :param bool cursor: whether to use the cursor position to go back after printing.
    """

    if cursor:  # Save cursor position:
        print(ESC + "[s", end='', flush=True)  # To get the cursor is necessary to flush.
    # Move to the desired position and print the text
    x = ESC + "[%d" % abs(x_cursor) + ('C' if x_cursor >= 0 else 'D')
    y = ESC + "[%d" % abs(y_cursor) + ('A' if y_cursor >= 0 else 'B')
    print(x + y + text, end='', flush=flush)
    if cursor:  # Go back to the initial cursor position
        print(ESC + "[u", end='', flush=flush)
    elif y_cursor > 0:
        print('\n' * y_cursor, end='', flush=flush)


def move_cursor_at(x_cursor=0, y_cursor=0, flush=True):
    """Move the cursor to the specify coordinates. Origin (0, 0) is the current
    cursor position. It uses ANSI escape (ESC) commands to move through the terminal's screen.
    See for example <https://es.wikipedia.org/wiki/C%C3%B3digo_escape_ANSI>`_ for additional details
    about these commands.

    :param int x_cursor: horizontal coordinate.
    :param int y_cursor: vertical coordinate.
    :param bool flush: whether to flush the stdout buffer or not (default: True).
    """
    # Move to the desired position
    x = ESC + "[%d" % abs(x_cursor) + ('C' if x_cursor >= 0 else 'D')
    y = ESC + "[%d" % abs(y_cursor) + ('A' if y_cursor >= 0 else 'B')
    print(x + y, end='', flush=flush)


def dict_depth(d):
    """Computes the depth of a dictionary recursively.
    :param dict d: dictionary.
    :return: depth.
    :rtype: int
    """
    if isinstance(d, dict):
        return 1 + (max(map(dict_depth, d.values())) if d else 0)
    return 0


def fmt_latex_table(tables, caption=('Caption of the table',), label=('tab:table1', ), **kwargs):
    """ Format the `tables` of parameters as latex tables.

    :param list of dict tables: tables of parameters as dictionaries. See :func:`lib_ring.generate_table`.
    :param list or tuple of str caption: captions of the tables.
    :param list or tuple of str label: labels of the tables.
    :param kwargs: other options to control the aspect of the tables.
    :return: list of tables, each composed of a list of strings that can be printed or saved to generate the latex
             code.
    :rytpe: [list of str]
    """

    shortcaption = kwargs.pop('shortcaption', [''] * len(tables))
    vem, wem = kwargs.pop('cwidths', (9, 3))
    header_def = kwargs.pop('header', [r"\it Symbol", "Value"])
    side_by_side = kwargs.pop('side_by_side', False)
    # Variables that affect the side-by-side layout
    table_sep = kwargs.pop('table_sep', 0.04)
    table_width = kwargs.pop('table_width', (1 - table_sep)/len(tables))
    scaletable = r"\textwidth" if side_by_side else kwargs.pop('scale', r"0.7\columnwidth")

    max_decimals = 3
    # ncols = len(header)

    # Write the header
    preamble = r"\begin{table}[htb]" + "\n"
    prending = r"    \bottomrule[0.8mm]" + "\n" + \
               r"  \end{tabular}}" + "\n"

    if len(tables) == 1 or not side_by_side:  # Single table or each table in different floating objects
        preamble = preamble + \
                   r"  \centering" + "\n"
        tab_prea = r"  \caption%s[%s]%s{\label{%s} %s}" + "\n" + \
                   r"  \vspace{2mm}" + "\n" + \
                   r"  \resizebox{%s}{!}{" + "\n" + \
                   r"  \begin{tabular}[t]{@{}v{%dem}w{%dem}r@{}}" + "\n" + \
                   r"    \toprule[0.8mm]" + "\n"
        prending = prending + r"  \vspace{-2mm}" + "\n"
    else:  # Multiple tables side by side
        tab_prea = r" \begin{minipage}[b]{%s\textwidth}" % table_width + "\n" + \
                   r" \centering" + "\n" + \
                   r"  \caption%s[%s]%s{\label{%s} %s}" + "\n" + \
                   r"  \resizebox{%s}{!}{" + "\n" + \
                   r"  \begin{tabular}[t]{@{}v{%dem}w{%dem}r@{}}" + "\n" + \
                   r"    \toprule[0.8mm]" + "\n"
        prending = prending + r" \end{minipage}" + "\n"

    ending = r"\end{table}" + "\n"

    header = r"     \multicolumn{1}{@{}v{%dem}}{%s}" + "\n" \
             r"     & {%s}" + "\n" \
             r"     & %s \\" + "\n" \
             r"    \cmidrule(r){1-2} \cmidrule(l){3-3}" + "\n"
    header = header % (vem, r"\sffamily \textbf{%s}", header_def[0], header_def[1])
    row = r"     \multicolumn{1}{v{%dem}}{%s}" + "\n" \
          r"     & %s" + "\n" \
          r"     & %s \\" + "\n"

    midrule = r"    \cmidrule(lr){1-1}" + "\n"
    blockrule = r"    \midrule[0.4mm]" + "\n"

    # Separation between tables for the side-by-side layout
    separator = r" \begin{minipage}[b]{%s\textwidth}" % table_sep + "\n" + \
                r"  \color{white} ." + "\n" + \
                r" \end{minipage}" + "\n"

    # Build the table
    latex_tables = [preamble] if side_by_side else []

    for t, table in enumerate(tables):
        latex_table = [] if side_by_side else [preamble]
        if t == 0 or not side_by_side:
            latex_table.append(tab_prea % ('', shortcaption[t], '', label[t], caption[t], scaletable, vem, wem))
        else:
            latex_table.append(separator)
            latex_table.append(tab_prea % ('of', shortcaption[t], r"{table}", label[t], caption[t], scaletable,
                                           vem, wem))
        for k, (key, v) in enumerate(table.items()):
            if k != 0:
                latex_table.append(blockrule)
            latex_table.append(header % v[0])
            for j, (key2, v2) in enumerate(v[1].items()):
                units = '' if v2[-1] == 'a.u.' else ' %s' % v2[-1]
                # Format the value according to its type
                value = v2[2]
                if isinstance(value, int):
                    value = f"${value:d}$"
                elif isinstance(value, tuple):  # This is interpreted as a closed range
                    vals = []
                    for val in value:
                        vals.append(format_float(val, max_decimals))
                    value = r"$\in \left[" + f"{vals[0]}, {vals[1]}" + r"\right]$"
                elif isinstance(value, Iterable) and not isinstance(value, str):
                    value = "$" + '$, $'.join([format_float(val, max_decimals) for val in value]) + "$"
                elif isinstance(value, float):
                    value = f"${format_float(value, max_decimals)}$"

                latex_table.append(row % (vem, v2[0], v2[1], ("%s" % value) + units))
                if j < len(v[1]) - 1:
                    latex_table.append(midrule)
        latex_table.append(prending)
        if not side_by_side:
            latex_table.append(ending)
            latex_tables.append(latex_table)
        else:
            latex_tables.extend(latex_table)
    if side_by_side:
        latex_tables.append(ending)
        latex_tables = [latex_tables]

    if kwargs.get('save', False):
        for t, latex_table in enumerate(latex_tables):
            filename = kwargs.get('filename', 'parameters_table')
            filename = filename + '.tex' if side_by_side else filename + ('_%02d-tab' % t)
            filename = check_overwrite(filename, auto=True, force=kwargs.pop('overwrite', False)).with_suffix('.tex')
            with open(filename, 'w') as fp:
                fp.writelines(latex_table)
                fp.close()

    return latex_tables


def truncate(number, digits):
    """ Truncate the decimal number up to the `digits` decimal.

    :param float number: decimal number to be truncated.
    :param int digits: number of decimals to conserve.
    :return: truncated decimal number
    :rtype: float
    """
    import math
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper


def format_float(x, max_decimals):
    import numpy as np
    value = x
    if isinstance(value, int):
        return value
    close = False
    decimals = -1
    while not close and decimals <= max_decimals:
        decimals += 1
        value = np.round(x, decimals=decimals)
        close = np.allclose(x, value)
    if close:  # Set the format if decimals <= `max_decimals`
        if decimals == 0:  # Integer
            value = f"{int(value):d}"
        elif value > 1 or decimals < 3 or truncate(value, 1) > 0:  # Normal float
            value = f"{value:.{decimals}f}"
        else:
            value = f"{int(value * 10**decimals):d}" + r"\cdot 10^{" + f"{-decimals}" + r"}"
    else:
        close = False
        m = 0
        value = x
        while not close and m < 100:
            m += 1
            value = int(np.round(m/x))
            close = np.allclose(m/x, value)
        if value < 0:
            m = -m
            value = -value
        value = f"{m}/{value}" if close else r"\sim %.*f" % (max_decimals, x)

    return value
