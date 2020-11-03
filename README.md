Overview
--------
Here we privode the new version of BBES: (new_bbes.py), with which we can get the traning data for  Graph Attention Network (GAT) model and test the model on both simulated&real_world data. Another file (GAT.ipynb) is the implemention of GAT training. 

Dependencies
----------
The script has been tested running under Python 3.6, with the following packages installed (along with their dependencies):
* dgl == 0.5.2
* numpy == 1.18.1
* scipy == 1.2.1
* pytorch ==1.5.1
* pandas ==1.0.3


new_bbes.py
-----------
Codes are mostly based on original version of van Ommen's BBES algorithm, except for the part that generates traning data for GAT and applies GAT model to a new branching heuristic. 

GAT.ipynb
---------
Contains data preprocessing, GAT building and GAT training. 
