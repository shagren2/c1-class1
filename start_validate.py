from decouple import config
from multiprocessing import cpu_count
from platform_classification.train import start_validate


def _start_validate():
    VAL_DATASET_PATH = config('VAL_DATASET_PATH')
    VAL_PATH_TO_CHECKPOINT_DIR = config('VAL_PATH_TO_CHECKPOINT_DIR')
    VAL_PATH_TO_SAVE_BEST = config('VAL_PATH_TO_SAVE_BEST')
    VAL_BATCH_SIZE = config('VAL_BATCH_SIZE', cast=int)
    VAL_NUM_WORKERS = config('VAL_NUM_WORKERS', cast=int)
    VAL_LOG_FREQUENCY = config('VAL_LOG_FREQUENCY', cast=int)
    VAL_USE_FILE_CACHE = config('VAL_USE_FILE_CACHE', default=False, cast=bool)

    if VAL_NUM_WORKERS == -1:
        VAL_NUM_WORKERS = cpu_count()

    start_validate(dataset_path=VAL_DATASET_PATH,
                   batch_size=VAL_BATCH_SIZE,
                   num_workers=VAL_NUM_WORKERS,
                   path_to_checkpoint_dir=VAL_PATH_TO_CHECKPOINT_DIR,
                   val_path_to_save_best=VAL_PATH_TO_SAVE_BEST,
                   print_freq=VAL_LOG_FREQUENCY,
                   use_file_cache=VAL_USE_FILE_CACHE)

if __name__ == '__main__':
    _start_validate()