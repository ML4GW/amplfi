export AMPLFI_DATADIR=~/amplfi/my-first-run/data/
export AMPLFI_CONDORDIR=~/amplfi/my-first-run/condor

LAW_CONFIG_FILE=./config.cfg poetry run law run amplfi.law.DataGeneration --workers 2 --dev
