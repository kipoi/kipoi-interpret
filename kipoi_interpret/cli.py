"""CLI
"""
from __future__ import absolute_import
from __future__ import print_function

import argparse
import logging
import os
import sys

from kipoi_interpret.importance_scores.cli import cli_feature_importance

import kipoi
from kipoi.cli.parser_utils import add_model, add_dataloader, file_exists, dir_exists
from kipoi import writers
from kipoi.utils import cd
from kipoi.utils import parse_json_file_str

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def isint(qstr):
    import re
    return bool(re.match("^[0-9]+$", qstr))


# Parse the slice
def parse_filter_slice(in_str):
    if in_str.startswith("(") or in_str.startswith("["):
        in_str_els = in_str.lstrip("([").rstrip(")]").split(",")
        slices = []
        for slice_el in in_str_els:
            slice_el = slice_el.strip(" ")
            if slice_el == "...":
                slices.append(Ellipsis)
            elif isint(slice_el):
                slices.append(int(slice_el))
                if len(in_str_els) == 1:
                    return int(slice_el)
            else:
                # taken from https://stackoverflow.com/questions/680826/python-create-slice-object-from-string
                slices.append(slice(*map(lambda x: int(x.strip()) if x.strip() else None, slice_el.split(':'))))
        return tuple(slices)
    elif isint(in_str):
        return int(in_str)
    else:
        raise Exception("Filter index slice not valid. Allowed values are e.g.: '1', [1:3,...], [:, 0:4, ...]")


def cli_grad(command, raw_args):
    """CLI interface to predict
    """
    # from .main import prepare_batch
    from kipoi.model import GradientMixin
    assert command == "grad"
    from tqdm import tqdm
    parser = argparse.ArgumentParser('kipoi {}'.format(command),
                                     description='Save gradients and inputs to a hdf5 file.')
    add_model(parser)
    add_dataloader(parser, with_args=True)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size to use in prediction')
    parser.add_argument("-n", "--num_workers", type=int, default=0,
                        help="Number of parallel workers for loading the dataset")
    parser.add_argument("-i", "--install_req", action='store_true',
                        help="Install required packages from requirements.txt")
    parser.add_argument("-l", "--layer", default=None,
                        help="Which output layer to use to make the predictions. If specified," +
                             "`model.predict_activation_on_batch` will be invoked instead of `model.predict_on_batch`",
                        required=False)
    parser.add_argument("--final_layer",
                        help="Alternatively to `--layer` this flag can be used to indicate that the last layer should "
                             "be used.", action='store_true')
    parser.add_argument("--pre_nonlinearity",
                        help="Flag indicating that it should checked whether the selected output is post activation "
                             "function. If a non-linear activation function is used attempt to use its input. This "
                             "feature is not available for all models.", action='store_true')
    parser.add_argument("-f", "--filter_idx",
                        help="Filter index that should be inspected with gradients. If not set all filters will " +
                             "be used.", default=None)
    parser.add_argument("-a", "--avg_func",
                        help="Averaging function to be applied across selected filters (`--filter_idx`) in " +
                             "layer `--layer`.", choices=GradientMixin.allowed_functions, default="sum")
    parser.add_argument('--selected_fwd_node', help="If the selected layer has multiple inbound connections in "
                                                    "the graph then those can be selected here with an integer "
                                                    "index. Not necessarily supported by all models.",
                        default=None, type=int)
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
    # install args
    if args.install_req:
        kipoi.pipeline.install_model_requirements(args.model,
                                                  args.source,
                                                  and_dataloaders=True)

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

    if not isinstance(model, GradientMixin):
        raise Exception("Model does not support gradient calculation.")

    if args.dataloader is not None:
        Dl = kipoi.get_dataloader_factory(args.dataloader, args.dataloader_source)
    else:
        Dl = model.default_dataloader

    dataloader_kwargs = kipoi.pipeline.validate_kwargs(Dl, dataloader_kwargs)
    dl = Dl(**dataloader_kwargs)

    filter_idx_parsed = None
    if args.filter_idx is not None:
        filter_idx_parsed = parse_filter_slice(args.filter_idx)

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
        pred_batch = model.input_grad(batch['inputs'], filter_idx=filter_idx_parsed,
                                      avg_func=args.avg_func, layer=layer, final_layer=args.final_layer,
                                      selected_fwd_node=args.selected_fwd_node,
                                      pre_nonlinearity=args.pre_nonlinearity)

        # write out the predictions, metadata (, inputs, targets)
        # always keep the inputs so that input*grad can be generated!
        # output_batch = prepare_batch(batch, pred_batch, keep_inputs=True)
        output_batch = batch
        output_batch["grads"] = pred_batch
        for writer in use_writers:
            writer.batch_write(output_batch)

    for writer in use_writers:
        writer.close()
    logger.info('Done! Gradients stored in {0}'.format(",".join(args.output)))


def cli_gr_inp_to_file(command, raw_args):
    """ CLI to save seq inputs of grad*input to a bigwig file
    """
    assert command == "gr_inp_to_file"
    parser = argparse.ArgumentParser('kipoi interpret {}'.format(command),
                                     description='Save grad*input in a file.')
    add_model(parser)
    add_dataloader(parser, with_args=True)
    parser.add_argument('-f', '--input_file', required=False,
                        help="Input HDF5 file produced from `grad`")
    parser.add_argument('-o', '--output', required=False,
                        help="Output bigwig for bedgraph file")
    parser.add_argument('--sample', required=False, type=int, default=None,
                        help="Input line for which the BigWig file should be generated. If not defined all"
                             "samples will be written.")
    parser.add_argument('--model_input', required=False, default=None,
                        help="Model input name to be used for plotting. " +
                        "As defined in model.yaml. Can be omitted if" +
                        "model only has one input.")
    args = parser.parse_args(raw_args)

    # Check that all the folders exist
    dir_exists(os.path.dirname(args.output), logger)
    # --------------------------------------------
    # install args
    from kipoi_interpret.gradviz import GradPlotter
    from kipoi.writers import BedGraphWriter

    logger.info('Loading gradient results file and model info...')

    gp = GradPlotter.from_hdf5(args.input_file, model=args.model, source=args.source)

    if args.sample is not None:
        samples = [args.sample]
    else:
        samples = list(range(gp.get_num_samples(args.model_input)))

    if args.output.endswith(".bed") or args.output.endswith(".bedgraph"):
        of_obj = BedGraphWriter(args.output)
    else:
        raise Exception("Output file format not supported!")

    logger.info('Writing...')

    for sample in samples:
        gp.write(sample, model_input=args.model_input, writer_obj=of_obj)

    logger.info('Saving...')

    of_obj.close()

    logger.info('Successfully wrote grad*input to file.')


# --------------------------------------------
# CLI commands


command_functions = {
    'feature_importance': cli_feature_importance,

    # Deprecate
    # 'gr_inp_to_file': cli_gr_inp_to_file,  # TODO - rename this to grad_input?
    # 'ism': cli_create_mutation_map,
}
commands_str = ', '.join(command_functions.keys())

# TODO - shall we call the grad commands: saliency?
parser = argparse.ArgumentParser(
    description='Kipoi model-zoo command line tool',
    usage='''kipoi interepret <command> [-h] ...

    # Available sub-commands:
    feature_importance    Compute the feature importance
    ''')
parser.add_argument('command', help='Subcommand to run; possible commands: {}'.format(commands_str))

CLI_DESCRIPTION = "Compute feature importance scores"


def cli_main(command, raw_args):
    args = parser.parse_args(raw_args[0:1])
    if args.command not in command_functions:
        parser.print_help()
        parser.exit(
            status=1,
            message='\nCommand `{}` not found. Possible commands: {}\n'.format(
                args.command, commands_str))
    command_fn = command_functions[args.command]
    command_fn(args.command, raw_args[1:])


if __name__ == '__main__':
    command = sys.argv[1]
    raw_args = sys.argv[1:]
    cli_main(command, raw_args)
