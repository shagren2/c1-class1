from decouple import config
from multiprocessing import cpu_count
from platform_classification.train import start_train


def _start_train():
    TRAIN_DATASET_PATH = config('TRAIN_DATASET_PATH')
    TRAIN_BATCH_SIZE = config('TRAIN_BATCH_SIZE', cast=int)
    TRAIN_NUM_WORKERS = config('TRAIN_NUM_WORKERS', cast=int)
    LEARNING_RATE = config('LEARNING_RATE', cast=float)
    BASE_MODEL_NAME = config('BASE_MODEL_NAME')
    NUM_EPOCH = config('NUM_EPOCH', cast=int)
    TRAIN_LOG_FREQUENCY = config('TRAIN_LOG_FREQUENCY', cast=int)
    SAVE_EACH_STEPS = config('SAVE_EACH_STEPS', cast=int)
    TRAIN_PATH_TO_SAVE_RESULT = config('TRAIN_PATH_TO_SAVE_RESULT')
    TRAIN_PATH_TO_INIT = config('TRAIN_PATH_TO_INIT', default=None)
    TRAIN_USE_FILE_CACHE = config('TRAIN_USE_FILE_CACHE', default=False, cast=bool)
    TRAIN_MIN_LOSS = config('TRAIN_MIN_LOSS', default=0.01, cast=float)

    if TRAIN_NUM_WORKERS == -1:
        TRAIN_NUM_WORKERS = cpu_count()


    start_train(dataset_path=TRAIN_DATASET_PATH,
                batch_size=TRAIN_BATCH_SIZE,
                num_workers=TRAIN_NUM_WORKERS,
                learning_rate=LEARNING_RATE,
                base_model=BASE_MODEL_NAME,
                num_epoch=NUM_EPOCH,
                log_frequ=TRAIN_LOG_FREQUENCY,
                save_each_step=SAVE_EACH_STEPS,
                path_to_save=TRAIN_PATH_TO_SAVE_RESULT,
                train_path_to_init=TRAIN_PATH_TO_INIT,
                use_file_cache=TRAIN_USE_FILE_CACHE,
                min_loss=TRAIN_MIN_LOSS)

if __name__ == '__main__':
    _start_train()