# Porous Equivariant Networks

Code for the paper: "[Equivariant Crystal Networks for Porous Materials](https://arxiv.org/abs/2304.01628)"

To train a network using our models and dataset, the following command should be called:

```
python code/train/train.py -z MFI -i 0 -r 1 -m pore -p 1 -a True -q True -n 200 -s 12
```

To use different models, ```-m``` should be either ```pore```, ```equi```, ```megnet```, ```dime```, ```schnet``` or ```cgcnn```.
To save the models with a different initial model number, change ```-i```.
To perform multiple training runs, change ```-r```. At each run, the parameters of the model are reinitialized.
To use a different proportion of the training set, change ```-p```.
In case ```-q``` is set to ```True```, a random training/testing split is used. Otherwise, all zeolites with at least ```-s``` aluminium atoms will be used as the test set.
With ```-n``` you can control the amount of epochs.

To switch material, change ```-z``` to either ```MOR``` or ```MFI``` (currently no other Zeolites/Materials are supported). For additional materials, the use should create their own (porous) representation of the material. 

In case you wish to use different hyperparameters, the model initialization should be changed in ```code/train/train.py```.
