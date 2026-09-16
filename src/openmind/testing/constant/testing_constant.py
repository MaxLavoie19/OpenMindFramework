#: Where a test run saves its tests' logs, under the run's root directory.
LOG_DIRECTORY = ("data", "log")
LOG_FORMAT = "%(asctime)s %(levelname)-5s %(name)s %(message)s"
#: The marker setting the lowest level a test saves: @pytest.mark.log_level("INFO").
LOG_LEVEL_MARKER = "log_level"
