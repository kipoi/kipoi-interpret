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
from kipoi.utils import parse_json_file_str, load_module
import numpy as np

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


def cli_deeplift(command, raw_args):
    """CLI interface to predict
    """
    # TODO: find a way to define the "reference" for a scored sequence.
    # from .main import prepare_batch
    assert command == "deeplift"
    from tqdm import tqdm
    from .referencebased import DeepLift
    from .referencebased import get_mxts_modes
    parser = argparse.ArgumentParser('kipoi interpret {}'.format(command),
                                     description='Calculate DeepLIFT scores.')
    add_model(parser)
    add_dataloader(parser, with_args=True)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to use in prediction')
    parser.add_argument("-n", "--num_workers", type=int, default=0,
                        help="Number of parallel workers for loading the dataset")
    parser.add_argument("-l", "--layer", type=int, default=None,
                        help="With respect to which layer the scores should be calculated.",
                        required=True)
    parser.add_argument("--pre_nonlinearity",
                        help="Flag indicating that it should checked whether the selected output is post activation "
                             "function. If a non-linear activation function is used attempt to use its input. This "
                             "feature is not available for all models.", action='store_true')
    parser.add_argument("-f", "--filter_idx",
                        help="Filter index that should be inspected with gradients", default=None, required=True,
                        type=int)
    parser.add_argument("-m", "--mxts_mode", help="Deeplift score, allowed values are: %s" % str(
        list(get_mxts_modes().keys())), default='rescale_conv_revealcancel_fc')
    parser.add_argument('-o', '--output', required=True, nargs="+",
                        help="Output files. File format is inferred from the file path ending. Available file formats are: " +
                             ", ".join(["." + k for k in writers.FILE_SUFFIX_MAP]))
    args = parser.parse_args(raw_args)

    dataloader_kwargs = parse_json_file_str(args.dataloader_args)

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

    layer = args.layer
    if layer is None and not args.final_layer:
        raise Exception("A layer has to be selected explicitely using `--layer` or implicitely by using the"
                        "`--final_layer` flag.")

    # Not a good idea
    # if layer is not None and isint(layer):
    #    logger.warn("Interpreting `--layer` value as integer layer index!")
    #    layer = int(args.layer)

    # load model & dataloader
    model = kipoi.get_model(args.model, args.source)

    if args.dataloader is not None:
        Dl = kipoi.get_dataloader_factory(args.dataloader, args.dataloader_source)
    else:
        Dl = model.default_dataloader

    dataloader_kwargs = kipoi.pipeline.validate_kwargs(Dl, dataloader_kwargs)
    dl = Dl(**dataloader_kwargs)

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

    d = DeepLift(model, output_layer=args.layer, task_idx=args.filter_idx,
                 preact=args.pre_nonlinearity, mxts_mode=args.mxts_mode,
                 batch_size=args.batch_size)

    # Loop through the data, make predictions, save the output
    for i, batch in enumerate(tqdm(it)):
        # validate the data schema in the first iteration
        if i == 0 and not Dl.output_schema.compatible_with_batch(batch):
            logger.warn("First batch of data is not compatible with the dataloader schema.")

        # calculate scores without reference for the moment.
        pred_batch = d.score(batch['inputs'], None)

        # write out the predictions, metadata (, inputs, targets)
        # always keep the inputs so that input*grad can be generated!
        # output_batch = prepare_batch(batch, pred_batch, keep_inputs=True)
        output_batch = batch
        output_batch["scores"] = pred_batch
        for writer in use_writers:
            writer.batch_write(output_batch)

    for writer in use_writers:
        writer.close()
    logger.info('Done! Gradients stored in {0}'.format(",".join(args.output)))


def cli_ism(command, raw_args):
    # TODO: find a way to define the model output selection
    """CLI interface to predict
    """
    # from .main import prepare_batch
    assert command == "ism"
    from tqdm import tqdm
    from .ism import Mutation

    parser = argparse.ArgumentParser('kipoi interpret {}'.format(command),
                                     description='Calculate DeepLIFT scores.')
    add_model(parser)
    add_dataloader(parser, with_args=True)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to use in prediction')
    parser.add_argument("-n", "--num_workers", type=int, default=0,
                        help="Number of parallel workers for loading the dataset")
    parser.add_argument("--model_input", help="Name of the model input that should be scored.", required=True)
    parser.add_argument('-s', "--scores", default="diff", nargs="+",
                        help="Scoring method to be used. Only scoring methods selected in the model yaml file are"
                             "available except for `diff` which is always available. Select scoring function by the"
                             "`name` tag defined in the model yaml file.")
    parser.add_argument('-k', "--score_kwargs", default=None, nargs="+",
                        help="JSON definition of the kwargs for the scoring functions selected in --scores. The "
                             "definiton can either be in JSON in the command line or the path of a .json file. The "
                             "individual JSONs are expected to be supplied in the same order as the labels defined in "
                             "--scores. If the defaults or no arguments should be used define '{}' for that respective "
                             "scoring method.")
    parser.add_argument("-c", "--category_axis", help="Using the selected model input with `--model_input`: Which "
                                                     "dimension of that array contains the one-hot encoded categories?"
                                                      " e.g. for a one-hot encoded DNA-sequence"
                                                     "array with input shape (1000, 4) for a single sample, "
                                                      "`category_dim` is 1, for (4, 1000) `category_dim`"
                                                     "is 0.", default=1, type =int, required=False)
    parser.add_argument("-f", "--output_sel_fn", help="Define an output selection function in order to return effects"
                                                      "on the output of the function. example definitoin: "
                                                      "`--output_sel_fn my_file.py::my_sel_fn`", default=None,
                                                      required = False)
    parser.add_argument('-o', '--output', required=True, nargs="+",
                        help="Output files. File format is inferred from the file path ending. "
                             "Available file formats are: " +
                             ", ".join(["." + k for k in writers.FILE_SUFFIX_MAP]))
    args = parser.parse_args(raw_args)

    dataloader_kwargs = parse_json_file_str(args.dataloader_args)

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
    if not isinstance(args.scores, list):
        args.scores = [args.scores]


    # load model & dataloader
    model = kipoi.get_model(args.model, args.source)

    if args.dataloader is not None:
        Dl = kipoi.get_dataloader_factory(args.dataloader, args.dataloader_source)
    else:
        Dl = model.default_dataloader

    dataloader_kwargs = kipoi.pipeline.validate_kwargs(Dl, dataloader_kwargs)
    dl = Dl(**dataloader_kwargs)

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

    output_sel_fn = None
    if args.output_sel_fn is not None:
        file_path, obj_name = tuple(args.output_sel_fn.split("::"))
        output_sel_fn = getattr(load_module(file_path), obj_name)

    m = Mutation(model, args.model_input, scores=args.scores, score_kwargs=args.score_kwargs,
                 batch_size=args.batch_size, output_sel_fn=output_sel_fn, category_axis=args.category_axis,
                 test_ref_ref = True)

    out_batches = {}

    # Loop through the data, make predictions, save the output..
    # TODO: batch writer fails because it tries to concatenate on highest dimension rather than the lowest!
    for i, batch in enumerate(tqdm(it)):
        # validate the data schema in the first iteration
        if i == 0 and not Dl.output_schema.compatible_with_batch(batch):
            logger.warn("First batch of data is not compatible with the dataloader schema.")

        # calculate scores without reference for the moment.
        pred_batch = m.score(batch['inputs'])

        # with the current writers it's not possible to store the scores and the model inputs in the same file
        output_batch = {}
        output_batch["scores"] = pred_batch

        for k in output_batch:
            if k not in out_batches:
                out_batches[k] = []
            out_batches[k].append(output_batch[k])

    # concatenate batches:
    full_output = {k: np.concatenate([np.array(el) for el in v])for k,v in out_batches.items()}
    logger.info('Full output shape: {0}'.format(str(full_output["scores"].shape)))

    for writer in use_writers:
        writer.batch_write(full_output)

    for writer in use_writers:
        writer.close()
    logger.info('Done! ISM stored in {0}'.format(",".join(args.output)))

