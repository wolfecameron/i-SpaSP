# i-SpaSP: Structured Neural Pruning via Sparse Signal Recovery


This is a public code repository for the publication:
> [**i-SpaSP: Structured Neural Pruning via Sparse Signal Recovery**]()<br>
> Cameron R Wolfe, Anastasios Kyrillidis<br>


## Environment/Dependencies

Requires anaconda to be installed (python3)
Anaconda can be installed at https://www.anaconda.com/products/individual

```bash
conda create -n ispasp python=3.6 anaconda
conda activate ispasp
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

## File Arrangement

Here we summarize all files present in this repo and their purpose.
```
+-- prune_resnet34_ispasp.py : i-SpaSP implementation for ResNet34
+-- prune_mobilenetv2_ispasp.py : i-SpaSP implementation for MobileNetV2
+-- lib/ : utility/helper functions
|   +-- data.py: helper functions for handling data
|   +-- utils.py: helper functions for computing performance metrics
+-- scripts/ : contains python scripts for running pruning experiments
|   +-- prune_rn34_ispasp.py: run an i-SpaSP pruning experiment for ResNet34  
|   +-- prune_mbnv2_ispasp.py: run an i-SpaSP pruning experiment for MobileNetV2
+-- requirements.txt : dependencies for pruning experiments
```
