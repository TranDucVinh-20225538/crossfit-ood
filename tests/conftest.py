import numpy as np
import pytest

from crossfit_ood import crossfit_auroc


def run(data, **kw):
    """crossfit_auroc on a GroupedData object."""
    kw.setdefault("labels_train", data.labels_train)
    kw.setdefault("groups_ood", data.groups_ood)
    kw.setdefault("uncertainty", "bootstrap")  # fast default for tests; others passed explicitly
    return crossfit_auroc(data.features_train, data.groups_train, data.features_id_eval,
                          data.groups_id_eval, data.features_ood, **kw)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
