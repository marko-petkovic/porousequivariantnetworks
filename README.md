# Porous Equivariant Networks

Code for the paper: "Equivariant Crystal Networks for Porous Materials"

To train a network using our models and dataset, the following command should be called:

```
python code/train/train.py -i 0 -r 1 -m pore -p 1
```

To use different models, ```-m``` should be either ```pore```, ```equi```, ```megnet```, ```schnet``` or ```cgcnn```.
To save the models with a different initial model number, change ```-i```.
To perform multiple training runs, change ```-r```.
To use a different proportion of the training set, change ```-p```.

In case you wish to use different hyperparameters, the model initialization should be changed in ```code/train/train.py```.
