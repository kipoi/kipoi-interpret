import pytest
import subprocess
import sys
import os
import yaml
import pandas as pd
import config
# import filecmp
# from utils import compare_vcfs
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


@pytest.mark.parametrize("example", list(predict_activation_layers))
def test_grad_predict_example(example, tmpdir):
    """kipoi grad ...
    """
    if example in {"rbp", "non_bedinput_model", "iris_model_template"} and sys.version_info[0] == 2:
        pytest.skip("rbp example not supported on python 2 ")

    example_dir = "tests/models/{0}".format(example)

    for file_format in ["tsv", "hdf5"]:
        print(example)

        layer_args = ["--layer", predict_activation_layers[example], ]
        final_layer_arg = ["--final_layer"]

        for i, la in enumerate([layer_args, final_layer_arg]):

            tmpfile = os.path.join(str(tmpdir), str("./grad_outputs.{0}.{1}".format(i, file_format)))
            # run the
            args = ["python", os.path.abspath("./kipoi_interpret/cli.py"),
                    "grad",
                    "../",  # directory
                    "--source=dir",
                    "--batch_size=4",
                    "--dataloader_args=test.json",
                    "--output", tmpfile]

            if i == 0 and INSTALL_FLAG:
                args.append(INSTALL_FLAG)
            if os.path.exists(tmpfile):
                os.unlink(tmpfile)
            returncode = subprocess.call(args=args + la, cwd=os.path.realpath(example_dir + "/example_files"))
            assert returncode == 0

            # Circle-ci is rediciulous about this
            # assert os.path.exists(tmpfile)

            if os.path.exists(tmpfile):
                os.unlink(tmpfile)

            continue


def test_deeplift_predict_example(tmpdir):
    """kipoi grad ...
    """
    example = "tal1_model"

    example_dir = "tests/models/{0}".format(example)
    os.system("pip install keras==2.0.9")

    for file_format in ["tsv", "hdf5"]:
        print(example)
        tmpfile = os.path.join(str(tmpdir), str("./grad_outputs.{0}".format(file_format)))

        # run the
        args = ["python", os.path.abspath("./kipoi_interpret/cli.py"),
                "deeplift",
                "../",  # directory
                "--source=dir",
                "--batch_size=4",
                "--dataloader_args=test.json",
                "--filter_idx=0",
                "--layer=-2",
                "--output", tmpfile]

        if INSTALL_FLAG:
            args.append(INSTALL_FLAG)

        if os.path.exists(tmpfile):
            os.unlink(tmpfile)

        returncode = subprocess.call(args=args, cwd=os.path.realpath(example_dir + "/example_files"))
        assert returncode == 0

        if os.path.exists(tmpfile):
            os.unlink(tmpfile)


@pytest.mark.parametrize("example", list(predict_activation_layers))
@pytest.mark.parametrize("use_output_sel", [False, True])
def test_ism_predict_example(example, use_output_sel, tmpdir):
    """kipoi grad ...
    """
    print("A")
    if example in {"rbp", "non_bedinput_model", "iris_model_template"} and sys.version_info[0] == 2:
        pytest.skip("rbp example not supported on python 2 ")

    print("B")
    example_dir = "tests/models/{0}".format(example)
    if example == "rbp":
        model_input_name = "seq"
    else:
        model_input_name = "input"

    print("C")
    for file_format in ["tsv", "hdf5"]:
        print('file_format',file_format)
        tmpfile = os.path.join(str(tmpdir), str("./grad_outputs.{0}".format(file_format)))

        # run the
        args = ["python", os.path.abspath("./kipoi_interpret/cli.py"),
                "ism",
                "../",  # directory
                "--source=dir",
                "--batch_size=4",
                "--model_input=" + model_input_name,
                "--dataloader_args=test.json",
                "--output", tmpfile]

        print("D")
        if use_output_sel:
            args.append("--output_sel_fn=out_sel_fn.py::sel")

        print("E")
        if INSTALL_FLAG:
            args.append(INSTALL_FLAG)

        print("F")
        if os.path.exists(tmpfile):
            os.unlink(tmpfile)

        print("G")
        returncode = subprocess.call(args=args, cwd=os.path.realpath(example_dir + "/example_files"))
        assert returncode == 0

        print("H")
        if os.path.exists(tmpfile):
            os.unlink(tmpfile)
