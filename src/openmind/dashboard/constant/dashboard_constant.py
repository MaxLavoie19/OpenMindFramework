#: Where and how often the dashboard serves its page by default.
DEFAULT_PORT = 8765
DEFAULT_REFRESH_SECONDS = 30

#: How many of the latest notable log lines, and of earlyoom's latest kills, the page shows.
RECENT_LINES = 15
EARLYOOM_LINES = 10

#: The loggers whose INFO and WARNING lines tell where a training is, as the training log writes their names.
NOTABLE_LOGGERS = (
    "__main__",
    "openmind.training.service.value_training_loop",
    "openmind.training.service.value_distiller",
    "openmind.training.service.position_ponderer",
    "openmind.inference.service.expression_search",
    "openmind.rbs.service.value_generator",
    "openmind.parallel.service.task_runner",
    "openmind.parallel.service.memory_guard",
)
ROUND_LOGGER = "openmind.training.service.value_training_loop"
GAME_LOGGER = "openmind.training.service.self_play"
MATCH_LOGGER = "openmind.evaluation.service.match_runner"
SEARCH_LOGGER = "openmind.mcts.service.tree_search"
DEDUCTION_LOGGER = "openmind.inference.service.position_deducer"

#: How a training's processes are recognized from their command lines: the training entrypoint, the script looping
#: over trainings, and the worker processes a training starts.
TRAINING_MODULE = "openmind.entrypoint.train_values"
LOOP_SCRIPT = "continue_training"
WORKER_MARK = "multiprocessing.spawn"

#: The roles a training process can have on the page.
LOOP, TRAINING, WORKER = "loop", "training", "worker"
