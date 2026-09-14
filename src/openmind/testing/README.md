# testing

## Purpose

Test support for OpenMind and for the projects built on it. Its pytest plugin saves each test's logs, so a test run
can be reviewed afterwards, in OpenMind and in a problem project alike.

## Content

| File | What it is |
|---|---|
| `plugin/log_saving.py` | A pytest plugin: registers the `log_level` marker and saves each test's logs in `<root>/data/log/<test file>/<test name>.log`, `<root>` being the test run's root directory |
| `constant/testing_constant.py` | The log directory (`data/log`), the log line format, and the marker's name |

## Usage

In a project's root `conftest.py`:

```python
pytest_plugins = ["openmind.testing.plugin.log_saving"]
```

Every test then saves its logs, from DEBUG up, as `LEVEL logger message` lines; a test marked
`@pytest.mark.log_level("INFO")` saves only INFO and above. A test that logs nothing saves no file.

## Notes

- Tests: `plugin/log_saving_tests.py`, which runs an example test file with pytest's `pytester`.
