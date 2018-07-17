import pytest
import subprocess
import sys
import os
import yaml
import pandas as pd
import config
# import filecmp
from utils import compare_vcfs
from kipoi.readers import HDF5Reader
import numpy as np

predict_activation_layers = {
    "rbp": "concatenate_6",
    "pyt": "3"  # two before the last layer
}

grad_inputs = {
    "rbp": "seq",
    "pyt": None
}

if config.install_req:
    INSTALL_FLAG = "--install_req"
else:
    INSTALL_FLAG = ""


def test_parse_filter_slice():
    from kipoi_interpret.cli import parse_filter_slice

    class DummySlice():

        def __getitem__(self, key):
            return key

    assert DummySlice()[1] == parse_filter_slice("[1]")
    assert DummySlice()[::-1, ...] == parse_filter_slice("[::-1,...]")
    assert DummySlice()[..., 1:3, :7, 1:, ...] == parse_filter_slice("[..., 1:3, :7, 1:, ...]")
    assert DummySlice()[..., 1:3, :7, 1:, ...] == parse_filter_slice("(..., 1:3, :7, 1:, ...)")
    assert DummySlice()[1] == parse_filter_slice("1")
    with pytest.raises(Exception):
        parse_filter_slice("[:::2]")


# @pytest.mark.parametrize("example", list(predict_activation_layers))
# def test_grad_predict_example(example):
#     """kipoi grad ...
#     """
#     if example in {"rbp", "non_bedinput_model", "iris_model_template"} and sys.version_info[0] == 2:
#         pytest.skip("rbp example not supported on python 2 ")

#     example_dir = "examples/{0}".format(example)

#     for file_format in ["tsv", "hdf5"]:
#         print(example)
#         tmpfile = os.path.realpath(str("./grad_outputs.{0}".format(file_format)))
#         bedgraph_temp_file = os.path.realpath(str("./grad_x_input.bed"))

#         # run the
#         args = ["python", "kipoi_interpret",
#                 "grad",
#                 "../",  # directory
#                 "--source=dir",
#                 "--batch_size=4",
#                 "--dataloader_args=test.json",
#                 "--output", tmpfile]
#         layer_args = ["--layer", predict_activation_layers[example], ]
#         final_layer_arg = ["--final_layer"]

#         if INSTALL_FLAG:
#             args.append(INSTALL_FLAG)

#         for la in [layer_args, final_layer_arg]:
#             returncode = subprocess.call(args=args + la, cwd=os.path.realpath(example_dir + "/example_files"))
#             assert returncode == 0

#             assert os.path.exists(tmpfile)

#             if file_format == "hdf5":
#                 data = HDF5Reader.load(tmpfile)
#                 assert {'metadata', 'grads', 'inputs'} <= set(data.keys())
#                 # Here we can attempt to write a bedgraph file:
#                 bg_args = ["python", "kipoi_interpret",
#                            "gr_inp_to_file",
#                            "../",  # directory
#                            "--source=dir",
#                            '--output', bedgraph_temp_file,
#                            "--input_file", tmpfile]
#                 if grad_inputs[example] is not None:
#                     bg_args += ["--model_input", grad_inputs[example]]
#                 returncode = subprocess.call(args=bg_args,
#                                              cwd=os.path.realpath(example_dir + "/example_files"))

#                 assert returncode == 0
#                 assert os.path.exists(bedgraph_temp_file)
#                 os.unlink(bedgraph_temp_file)

#             else:
#                 data = pd.read_csv(tmpfile, sep="\t")
#                 inputs_columns = data.columns.str.contains("inputs/")
#                 preds_columns = data.columns.str.contains("grads/")
#                 assert np.all(np.in1d(data.columns.values[preds_columns],
#                                       data.columns.str.replace("inputs/", "grads/").values[inputs_columns]))
#                 other_cols = data.columns.values[~(preds_columns | inputs_columns)]
#                 expected = ['metadata/ranges/chr',
#                             'metadata/ranges/end',
#                             'metadata/ranges/id',
#                             'metadata/ranges/start',
#                             'metadata/ranges/strand']
#                 assert np.all(np.in1d(expected, other_cols))

#             os.unlink(tmpfile)
