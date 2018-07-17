"""CLI
"""
from __future__ import absolute_import
from __future__ import print_function

import argparse
import logging
import os
import sys

#from kipoi_interpret.feature_importance import available_importance_scores, get_importance_score

from tqdm import tqdm
import kipoi
from kipoi.cli.parser_utils import add_model, add_dataloader, file_exists, dir_exists
from kipoi import writers
from kipoi.utils import parse_json_file_str

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def cli_feature_importance(command, raw_args):
    """CLI interface to predict
    """
    # from .main import prepare_batch
    assert command == "feature_importance"
    parser = argparse.ArgumentParser('kipoi {}'.format(command),
                                     description='Save gradients and inputs to a hdf5 file.')
    add_model(parser)
    add_dataloader(parser, with_args=True)
    parser.add_argument("--imp_score", help="Importance score name", choices=available_importance_scores())
    parser.add_argument("--imp_score_kwargs", help="Importance score kwargs")
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to use in prediction')
    parser.add_argument("-n", "--num_workers", type=int, default=0,
                        help="Number of parallel workers for loading the dataset")
    # TODO - handle the reference-based importance scores...

    # io
    parser.add_argument('-o', '--output', required=True, nargs="+",
                        help="Output files. File format is inferred from the file path ending. Available file formats are: " +
                             ", ".join(["." + k for k in writers.FILE_SUFFIX_MAP]))
    args = parser.parse_args(raw_args)

    dataloader_kwargs = parse_json_file_str(args.dataloader_args)
    imp_score_kwargs = parse_json_file_str(args.imp_score_kwargs)

    # setup the files
    if not isinstance(args.output, list):
        args.output = [args.output]
    for o in args.output:
        ending = o.split('.')[-1]
        if ending not in writers.FILE_SUFFIX_MAP:
            logger.error("File ending: {0} for file {1} not from {2}".
                         format(ending, o, writers.FILE_SUFFIX_MAP))
            sys.exit(1)
        dir_exists(os.path.dirname(o), logger)
    # --------------------------------------------
    # install args
    if args.install_req:
        kipoi.pipeline.install_model_requirements(args.model,
                                                  args.source,
                                                  and_dataloaders=True)

    # load model & dataloader
    model = kipoi.get_model(args.model, args.source, with_dataloader=args.dataloader is None)

    if args.dataloader is not None:
        Dl = kipoi.get_dataloader_factory(args.dataloader, args.dataloader_source)
    else:
        Dl = model.default_dataloader

    dataloader_kwargs = kipoi.pipeline.validate_kwargs(Dl, dataloader_kwargs)
    dl = Dl(**dataloader_kwargs)

    # get_importance_score
    ImpScore = get_importance_score(args.imp_score)
    if not ImpScore.is_compatible(model):
        raise ValueError("model not compatible with score: {0}".format(args.imp_score))
    impscore = ImpScore(model, **imp_score_kwargs)

    # setup batching
    it = dl.batch_iter(batch_size=args.batch_size,
                       num_workers=args.num_workers)

    # Setup the writers
    use_writers = []
    for output in args.output:
        ending = output.split('.')[-1]
        W = writers.FILE_SUFFIX_MAP[ending]
        logger.info("Using {0} for file {1}".format(W.__name__, output))
        if ending == "tsv":
            assert W == writers.TsvBatchWriter
            use_writers.append(writers.TsvBatchWriter(file_path=output, nested_sep="/"))
        elif ending == "bed":
            raise Exception("Please use tsv or hdf5 output format.")
        elif ending in ["hdf5", "h5"]:
            assert W == writers.HDF5BatchWriter
            use_writers.append(writers.HDF5BatchWriter(file_path=output))
        else:
            logger.error("Unknown file format: {0}".format(ending))
            sys.exit(1)

    # Loop through the data, make predictions, save the output
    for i, batch in enumerate(tqdm(it)):
        # validate the data schema in the first iteration
        if i == 0 and not Dl.output_schema.compatible_with_batch(batch):
            logger.warn("First batch of data is not compatible with the dataloader schema.")

        # make the prediction
        # TODO - handle the reference-based importance scores...
        importance_scores = impscore.score(batch['inputs'])

        # write out the predictions, metadata (, inputs, targets)
        # always keep the inputs so that input*grad can be generated!
        # output_batch = prepare_batch(batch, pred_batch, keep_inputs=True)
        output_batch = batch
        output_batch["importance_scores"] = importance_scores
        for writer in use_writers:
            writer.batch_write(output_batch)

    for writer in use_writers:
        writer.close()
    logger.info('Done! Importance scores stored in {0}'.format(",".join(args.output)))
