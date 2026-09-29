#!/usr/bin/env python
# coding: utf-8
from utils import *
from cellpose import models

config = UnifiedConfig()

train_data, test_data = prepare_data(config.dataset.base_directory)

ex_im = test_data[0][0]

cm_cfg = ModelBuildConfig()
cm_cfg.pretrained_model = "promising/Model-C00.pt"
cm = models.CellposeModel(**cm_cfg.todict())
nm_cfg = ModelBuildConfig()
nm_cfg.pretrained_model = "promising/Model-N00.pt"
nm = models.CellposeModel(**nm_cfg.todict())

tr_ma, _, _ = cm.eval(train_data[0])
te_ma, _, _ = nm.eval(test_data[0])
