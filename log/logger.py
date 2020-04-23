import logging
import config as cfg


class CustomLogger:
    def __init__(self, base_logger):
        self.logger = base_logger
        self.history = []

    def __call__(self, msg, do_print=True):
        self.history.append(msg)
        if do_print:
            print(msg)
        self.logger.info(msg)

    def write(self, writer):
        writer.add_text(f"Logging", '<br />'.join(self.history), 0)
        return writer


def get_logger(filepath):
    logger = logging.getLogger("Customlogger")
    logger.setLevel(logging.INFO)
    file = logging.FileHandler(filepath, mode='w')
    file.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file)
    return CustomLogger(logger)


logger = get_logger(cfg.LOG_FILE)
