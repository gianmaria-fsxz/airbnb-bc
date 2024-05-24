import logging
import subprocess
import sys
import warnings

log = logging.getLogger("INSTALLATION")


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    log.info("creating feature store")
    out = subprocess.call("feast init feature_store", shell=True)
    if out == 0:
        log.info("ok")
    else:
        sys.exit(1)

    log.info("removing useless default files")
    subprocess.call(
        "rm  ./feature_store/feature_repo/example_repo.py ./feature_store/feature_repo/test_workflow.py ./feature_store/feature_repo/data/driver_stats.parquet",
        shell=True,
    )

    log.info("moving fs definition")
    subprocess.call(
        "cp fs_definition.py feature_store/feature_repo/fs_definition.py", shell=True
    )

    code = subprocess.call("python prepare_data.py", shell=True)
    if code == 0:
        log.info("data preparation done!")
    else:
        log.info("error in creating data!")
        sys.exit(1)

    log.info("starting fs creation...")
    code = subprocess.call("cd feature_store/feature_repo && feast apply", shell=True)
    if code == 0:
        log.info("fs created!")
    else:
        log.info("error in creating fs!")
        sys.exit(1)

    log.info("starting training dataset generation")
    code = subprocess.call("python prepare_training.py", shell=True)
    if code == 0:
        log.info("training dataset created!")
    else:
        log.info("error in creating training dataset!")
        sys.exit(1)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
