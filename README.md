# Sleep-Model-Test

## Preliminary File Preparation
1) Convert HRV data from txt to csv using the "TXT-to-CSV" code provided.
2) Create mask around bed frame of participant using either a custom segmentation script or manually inputing corners of badframe into "Tensor-Creation" and create participant tensor.
3) Convert tensors from .pt format to .npz using "PT-to-NPZ" script.

## Using Model to Inference
1) Follow the steps outlines in the Jupyter notebook "Model-Test", making sure to edit the file paths for the HRV Model, Transformer Model

This repository is representative of the work carried out as part of a third year udergraduate dissertation on sleep TimeSformers by Andreas Tsouras.

© Andreas Tsouras, 2025. All rights reserved.

This repository and all source code, models, algorithms, and documentation contained within are the original work of Andreas Tsouras. The contents are shared for academic and evaluation purposes only.

No part of this repository may be reproduced, redistributed, modified, or used for commercial purposes without the express written permission of the author.

If you wish to collaborate, reproduce, or adapt any part of this work, please contact the author directly.
