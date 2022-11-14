"""
Author: Ivo Gollini Navarrete
Date: 21/august/2022
Institution: MBZUAI
"""

# IMPORTS
import sys
import argparse
import os
from utils import preprocess
from experiments.inf_seg import Inference
from experiments.inf_class import Inf_class


def path(string):
    if os.path.exists(string):
        return string
    else:
        sys.exit(f'File not found: {string}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('process', metavar='process', type=str, help='Process to be performed (Preprocess, train segmentation, inference)')
    parser.add_argument('input', metavar='input', type=path, help='Path to the input dataset')
    parser.add_argument('output', metavar='output', type=str, help='path to preprocessed data output / inference weights / inference classifiers')
    parser.add_argument('dataset', metavar='dataset', type=str, help='Select dataset / model to be used')
    parser.add_argument(
    "-D",
    "--debug",
    default=False,
    action="store_true",
    help="""Flag to set the experiment to debug mode.""")

    args = parser.parse_args()

    if args.process == "preprocess":
        data_preprocess = preprocess.Preprocess(args.input, args.output, args.dataset)
        if args.dataset == "radgen": data_preprocess.radiogenomics()
        elif args.dataset == "rad": data_preprocess.radiomics()
        elif args.dataset == "msd": data_preprocess.msd()

    elif args.process == "inf_seg":
        inference = Inference(args.input, args.output, args.dataset)
        inference.run()
    
    elif args.process == "inf_class":
        inf_class = Inf_class(args.input, args.output, args.dataset)
        inf_class.run()

if __name__ == "__main__":
    main()