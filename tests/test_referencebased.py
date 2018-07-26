from kipoi_interpret.importance_scores.referencebased import DeepLift
import kipoi
import pytest
import config

INSTALL_REQ = config.install_req
from kipoi.pipeline import install_model_requirements
import json
from tqdm import tqdm
from kipoi import writers, readers
import numpy as np
import os


predict_activation_layers = {
    "rbp": "concatenate_6",
    "tal1_model": "dense_1"
    # "pyt": "3"  # two before the last layer
}


def test_deeplift():
    # return True
    example = "tal1_model"
    layer = predict_activation_layers[example]
    example_dir = "tests/models/{0}".format(example)
    if INSTALL_REQ:
        install_model_requirements(example_dir, "dir", and_dataloaders=True)

    model = kipoi.get_model(example_dir, source="dir")
    # The preprocessor
    Dataloader = kipoi.get_dataloader_factory(example_dir, source="dir")
    #
    with open(example_dir + "/example_files/test.json", "r") as ifh:
        dataloader_arguments = json.load(ifh)

    for k in dataloader_arguments:
        dataloader_arguments[k] = "example_files/" + dataloader_arguments[k]

    d = DeepLift(model, output_layer=-2, task_idx=0, preact=None, mxts_mode='grad_times_inp')

    new_ofname = model.source_dir + "/example_files/deeplift_grads_pred.hdf5"
    if os.path.exists(new_ofname):
        os.unlink(new_ofname)

    writer = writers.HDF5BatchWriter(file_path=new_ofname)

    with kipoi.utils.cd(model.source_dir):
        dl = Dataloader(**dataloader_arguments)
        it = dl.batch_iter(batch_size=32, num_workers=0)
        # Loop through the data, make predictions, save the output
        for i, batch in enumerate(tqdm(it)):
            # make the prediction
            pred_batch = d.score(batch['inputs'], None)

            # Using Avanti's recommendation to check whether the model conversion has worked.
            pred_batch_fwd = d.predict_on_batch(batch['inputs'])
            orig_pred_batch_fwd = model.predict_on_batch(batch['inputs'])
            assert np.all(pred_batch_fwd == orig_pred_batch_fwd)

        output_batch = batch
        output_batch["input_grad"] = pred_batch
        writer.batch_write(output_batch)
    writer.close()

    new_res = readers.HDF5Reader.load(new_ofname)
    ref_res = readers.HDF5Reader.load(model.source_dir + "/example_files/grads.hdf5")
    assert np.all(np.isclose(new_res['input_grad'], (ref_res['inputs'] * ref_res['grads'])))

    if os.path.exists(new_ofname):
        os.unlink(new_ofname)
