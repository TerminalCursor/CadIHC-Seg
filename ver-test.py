#!/usr/bin/env python
# coding: utf-8
from cellpose import models

# Model Preparation
model = models.CellposeModel()
print(model.net)

model = models.CellposeModel(model_type="cyto")
print(model.net)
