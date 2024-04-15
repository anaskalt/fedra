====================================
Fedra - Federated Learning Framework
====================================


General Description
===================

**Fedra** is an innovative federated learning framework that enables decentralized training across distributed nodes. It leverages peer-to-peer (P2P) communication to facilitate model training on edge devices, allowing nodes to directly exchange model updates and perform local aggregation. This decentralized approach enhances privacy and security, as no raw data is shared and no central server is required. Fedra is particularly suitable for scenarios where data privacy is paramount.

Table of Contents
=================

.. contents::

Detailed Description
====================

Overview
--------

**Fedra** revolutionizes federated learning by employing a decentralized model. Instead of relying on a central aggregator, each node independently trains a model on its data and engages in mutual exchange of model updates with other nodes. Local aggregation is then performed on each node, integrating insights from across the network to refine the model. This process iterates until the collective model converges or a specified number of rounds is complete.

Configuration
=============

Setting up **Fedra** is streamlined and user-friendly. The framework requires node-specific configurations, managed via a simple configuration file (`node.conf`). This file encapsulates essential parameters such as model details, training rounds, and P2P network settings, ensuring a seamless initiation into the federated learning process.

Supported Models
================

The framework is model-agnostic, allowing for the integration of various neural network models. The default implementation includes a dense neural network specified in `models/net.py`, which can be easily replaced or extended based on the application requirements.

Federated Learning Process
==========================

1. **Initialization**: The global model is initialized, and configuration parameters are distributed to all participating nodes.
2. **Local Training**: Each node trains the model locally with its dataset.
3. **Model Update Exchange**: Post-training, nodes serialize and exchange model updates amongst themselves via P2P communication.
4. **Local Aggregation**: Each node deserializes received updates and performs local federated averaging to refine their model.
5. **Convergence and Repetition**: The iterative process continues, with each round of training and aggregation progressively enhancing the model's accuracy.

Repository Structure
====================

The **fedra** project is structured as follows:

- `fedra/main.py`: Entry point for the federated learning process.
- `fedra/models/`: Contains the neural network models.
- `fedra/network/`: Implements the P2P communication.
- `fedra/utils/`: Provides utility functions for data loading and weight manipulation.
- `data/`: Hosts the dataset used for training.
- `tests/`: Contains unit tests for various components of the project.
- `docs/`: Documentation for the project.

Generating Documentation
========================

To generate the documentation for **Fedra**, you first need to install the necessary dependencies:

.. code-block:: bash

    pip3 install -r requirements.txt

Once the dependencies are installed, you can compile the documentation using the following command:

.. code-block:: bash

    tox -e docs

This will generate HTML documentation in the `docs/_build/html/` directory. Ensure that you have `tox` installed in your environment to use this command.

Versioning
==========

The current version of the project is maintained in the `VERSION` file. Ensure to update this file as the project evolves.

License
=======

**fedra** is licensed under the MIT License. See the `LICENSE` file for more details.

Contributing
============

Contributions to **fedra** are welcome! Please refer to the `CONTRIBUTING.md` file for guidelines on how to contribute to this project.

About the Author
================

Developed by Anastasios Kaltakis, **fedra** reflects a dedication to advancing the field of federated learning with a focus on privacy-preserving techniques. With extensive experience in machine learning and software development, Anastasios has committed to creating a framework that empowers users to collaborate on machine learning tasks while maintaining the privacy of their data.
