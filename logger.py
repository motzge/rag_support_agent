import logging
from pathlib import Path



LOG_FILE: Path = Path(__file__).parent / "data" / "agent.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)



def get_logger(name: str) -> logging.logger:
    logger = logging.getLogger(name)


    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)


    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


    #Console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)


    #File
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)



    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


    return logger

