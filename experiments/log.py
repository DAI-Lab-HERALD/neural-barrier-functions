import errno
import logging.config
import os

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',  # Default is stderr
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': False
        },
        '__main__': {  # main logger
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': False
        },
        'experiments': {  # experiment logger
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}


def mkdir_p(path):
    """http://stackoverflow.com/a/600612/190597 (tzot)"""
    try:
        os.makedirs(path, exist_ok=True)  # Python>3.2
    except TypeError:
        try:
            os.makedirs(path)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise


class MakeFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=0):
        mkdir_p(os.path.dirname(filename))
        logging.FileHandler.__init__(self, filename, mode, encoding, delay)


def configure_logging(log_file):
    if log_file is not None:
        LOGGING_CONFIG['handlers']['file'] = {
                'level': 'DEBUG',
                'formatter': 'standard',
                'filename': log_file,
                'mode': 'a',
                'class': 'log.MakeFileHandler'
            }
        LOGGING_CONFIG['loggers']['']['handlers'].append('file')
        LOGGING_CONFIG['loggers']['__main__']['handlers'].append('file')
        LOGGING_CONFIG['loggers']['population']['handlers'].append('file')

    # Run once at startup:
    logging.config.dictConfig(LOGGING_CONFIG)

    # Include in each module:
    logger = logging.getLogger(__name__)
    logger.debug('Logging is configured.')
