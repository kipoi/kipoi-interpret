from __future__ import absolute_import

__author__ = 'Kipoi team'
__email__ = 'avsec@in.tum.de'
__version__ = '0.1.0'

# from .vis import GradPlotter

# Required by kipoi
from .cli import cli_main
# kipoi-interpret doesn' need any global variables
DataloaderParser = None
ModelParser = None
