#: How a log line is written: when it was produced, then its level, the service that wrote it and what it says. Runs
#: last for hours or days, and a line without its time can't be placed against anything else that happened.
LOG_FORMAT = "%(asctime)s %(levelname)-5s %(name)s %(message)s"
