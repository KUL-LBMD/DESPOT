#!/bin/bash

source ~/miniconda3/bin/activate
conda activate DESPOT

python scripts/train_despot.py --database HiQBind
python scripts/train_despot.py --database CROWN_train
python scripts/train_despot.py --database CROWN_xtal

python scripts/benchmark_casf.py --database HiQBind
python scripts/benchmark_casf.py --database CROWN_train
python scripts/benchmark_casf.py --database CROWN_xtal

python scripts/train_korp.py --database HiQBind
python scripts/train_korp.py --database CROWN_train
python scripts/train_korp.py --database CROWN_xtal

python scripts/benchmark_casf_korp.py --database HiQBind
python scripts/benchmark_casf_korp.py --database CROWN_train
python scripts/benchmark_casf_korp.py --database CROWN_xtal
