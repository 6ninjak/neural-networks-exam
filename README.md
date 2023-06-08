# PGExplainer
PGExplainer is an explainer model for graph neural networks that aims to generalize the prediction of a GNN explanation by training a deep neural network model on a set of instances instead of a single one.
The explanation of a GNN is expected to be the subgraph that contains only the edges that are relevant for GNN's prediction.

Knowing this, PGExplainer predicts explanations by maximizing the mutual information between the GNN predictions on the entire graph and the underlying structure (the predicted subgraph).

A more detailed explanation of the method is given in the notebook.

## Implementation
The implementation is done using [![PyTorch Lightning](https://img.shields.io/badge/pytorch-lightning-blue.svg?logo=PyTorch%20Lightning)](https://www.pytorchlightning.ai/index.html).

In order to run that code, you need to have installed Python (version 3.8) and then install all the required packages via the requirements.txt file in this repository running:

`pip install -r requirements.txt`

The code consists of two GNN models (one for node classification and one for graph classification) and the PGExplainer model. All the models can be trained from scratch on two datasets (BA-Shapes and a custom BA2-Motifs) or you can use the pretrained models that are provided in the repository.