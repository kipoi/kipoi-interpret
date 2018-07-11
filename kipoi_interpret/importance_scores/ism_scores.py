from __future__ import absolute_import
from __future__ import print_function

import copy
import warnings
import numpy as np
import abc
import logging
import six

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def default_kwargs(args):
    """Return the example kwargs
    """
    return {k: v.default for k, v in six.iteritems(args) if v.default is not None}


class Score(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, ref, alt, ref_rc=None, alt_rc=None):
        raise NotImplementedError("Analysis routine has to be implemented")

    @staticmethod
    def from_string(name):
        return scoring_options[name]


class RCScore(Score):
    allowed_str_opts = ["min", "max", "mean", "median", "absmax"]

    #

    def __init__(self, rc_merging="mean"):
        if isinstance(rc_merging, str):
            if rc_merging == "absmax":
                self.rc_merging = self.absmax
            elif rc_merging in self.allowed_str_opts:
                self.rc_merging = lambda x, y: getattr(np, rc_merging)([x, y], axis=0)
        elif callable(rc_merging):
            self.rc_merging = rc_merging
        else:
            raise Exception("rc_merging has to be a callable function of a string: %s" % str(self.allowed_str_opts))

    #

    def __call__(self, ref, alt, ref_rc=None, alt_rc=None):
        raise NotImplementedError("Analysis routine has to be implemented")

    @staticmethod
    def absmax(x, y, inplace=True):
        if not inplace:
            x = copy.deepcopy(x)
        replace_filt = np.abs(x) < np.abs(y)
        x[replace_filt] = y[replace_filt]
        return x


class Logit(RCScore):
    def __call__(self, ref, alt, ref_rc=None, alt_rc=None):
        preds = {"ref": ref, "ref_rc": ref_rc, "alt": alt, "alt_rc": alt_rc}
        if np.any([(preds[k].min() < 0 or preds[k].max() > 1) for k in preds if preds[k] is not None]):
            warnings.warn("Using log_odds on model outputs that are not bound [0,1]")
        diffs = np.log(preds["alt"] / (1 - preds["alt"])) - np.log(preds["ref"] / (1 - preds["ref"]))

        if (preds["ref_rc"] is not None) and ((preds["alt_rc"] is not None)):
            diffs_rc = np.log(preds["alt_rc"] / (1 - preds["alt_rc"])) - np.log(preds["ref_rc"] / (1 - preds["ref_rc"]))
            return self.rc_merging(diffs, diffs_rc)
        else:
            return diffs


class LogitAlt(RCScore):
    def __call__(self, ref, alt, ref_rc=None, alt_rc=None):
        preds = {"ref": ref, "ref_rc": ref_rc, "alt": alt, "alt_rc": alt_rc}
        if np.any([(preds[k].min() < 0 or preds[k].max() > 1) for k in preds if preds[k] is not None]):
            warnings.warn("Using log_odds on model outputs that are not bound [0,1]")
        logits = np.log(preds["alt"] / (1 - preds["alt"]))

        if preds["alt_rc"] is not None:
            logits_rc = np.log(preds["alt_rc"] / (1 - preds["alt_rc"]))
            return self.rc_merging(logits, logits_rc)
        else:
            return logits


class LogitRef(RCScore):
    def __call__(self, ref, alt, ref_rc=None, alt_rc=None):
        preds = {"ref": ref, "ref_rc": ref_rc, "alt": alt, "alt_rc": alt_rc}
        if np.any([(preds[k].min() < 0 or preds[k].max() > 1) for k in preds if preds[k] is not None]):
            warnings.warn("Using log_odds on model outputs that are not bound [0,1]")
        logits = np.log(preds["ref"] / (1 - preds["ref"]))

        if preds["ref_rc"] is not None:
            logits_rc = np.log(preds["ref_rc"] / (1 - preds["ref_rc"]))
            return self.rc_merging(logits, logits_rc)
        else:
            return logits


class Alt(RCScore):
    def __call__(self, ref, alt, ref_rc=None, alt_rc=None):
        alt_out = alt
        if alt_rc is not None:
            alt_out = self.rc_merging(alt, alt_rc)
        return alt_out


class Ref(RCScore):
    def __call__(self, ref, alt, ref_rc=None, alt_rc=None):
        ref_out = ref
        if ref_rc is not None:
            ref_out = self.rc_merging(ref, ref_rc)
        return ref_out


class Diff(RCScore):
    def __call__(self, ref, alt, ref_rc=None, alt_rc=None):
        preds = {"ref": ref, "ref_rc": ref_rc, "alt": alt, "alt_rc": alt_rc}
        diffs = preds["alt"] - preds["ref"]

        if (preds["ref_rc"] is not None) and ((preds["alt_rc"] is not None)):
            diffs_rc = preds["alt_rc"] - preds["ref_rc"]
            return self.rc_merging(diffs, diffs_rc)
        else:
            return diffs


class DeepSEA_effect(RCScore):
    def __call__(self, ref, alt, ref_rc=None, alt_rc=None):
        if ref_rc is None or alt_rc is None:
            raise Exception("DeepSEA_effect can only be computed if outputs for reverse complements are available.")
        preds = {"ref": ref, "ref_rc": ref_rc, "alt": alt, "alt_rc": alt_rc}
        if np.any([(preds[k].min() < 0 or preds[k].max() > 1) for k in preds if preds[k] is not None]):
            warnings.warn("Using log_odds on model outputs that are not bound [0,1]")
        logit_diffs = np.log(preds["alt"] / (1 - preds["alt"])) - np.log(preds["ref"] / (1 - preds["ref"]))
        diffs = preds["alt"] - preds["ref"]

        if (preds["ref_rc"] is not None) and ((preds["alt_rc"] is not None)):
            logit_diffs_rc = np.log(preds["alt_rc"] / (1 - preds["alt_rc"])) - np.log(
                preds["ref_rc"] / (1 - preds["ref_rc"]))
            diffs_rc = preds["alt_rc"] - preds["ref_rc"]
            logit_diffs = self.rc_merging(logit_diffs, logit_diffs_rc)
            diffs = self.rc_merging(diffs, diffs_rc)
            # self.rc_merging(np.abs(logit_diffs) * np.abs(diffs), np.abs(logit_diffs_rc) * np.abs(diffs_rc))

        return np.abs(logit_diffs) * np.abs(diffs)

builtin_default_kwargs = {"rc_merging": "mean"}

scoring_options = {
    "ref": Ref,
    "alt": Alt,
    "diff": Diff,
    "logit_ref": LogitRef,
    "logit_alt": LogitAlt,
    "logit": Logit,
    "deepsea_effect": DeepSEA_effect
}
