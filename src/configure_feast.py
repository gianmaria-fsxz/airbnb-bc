import subprocess

subprocess.call("feast init feature_store", shell=True)
subprocess.call("rm  ./feature_store/feature_repo/example_repo.py ./feature_store/feature_repo/test_workflow.py", shell=True)

subprocess.call("cp ./fs_definition.py ./feature_store/feature_repo/fs_definition.py ./feature_store/feature_repo/data/driver_stats.parquet", shell=True)
subprocess.call("cd feature_store/feature_repo && feast apply", shell=True)
