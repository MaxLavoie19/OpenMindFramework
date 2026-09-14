# Every test saves its logs in data/log/ (see src/openmind/testing/README.md); pytester runs the plugin's own tests.
pytest_plugins = ["openmind.testing.plugin.log_saving", "pytester"]
