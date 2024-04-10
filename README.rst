=====================================================
Anomaly Detection Federated Learning (anomaly-dfl) v1
=====================================================


General Description
===================

**anomaly-dfl** is a federated learning framework designed for anomaly detection across distributed nodes. It leverages local data processing to train machine learning models on edge devices, aggregating the model updates in a central server while preserving data privacy. This project facilitates collaborative learning without sharing raw data, making it suitable for privacy-sensitive applications.

Table of Contents
=================

.. contents::

Detailed Description
====================

Overview
--------

**anomaly-dfl** operates by distributing the model training process across multiple nodes. Each node trains a local model based on its dataset and shares the model updates with a central aggregator. The aggregator then performs federated averaging to update the global model. This cycle repeats over several rounds until the model converges or a predetermined number of rounds is completed.

Configuration
=============

The configuration for **anomaly-dfl** is straightforward, requiring only the setup of node-specific parameters and network configurations. Configuration can be achieved through a simple configuration file (`node.conf`), which specifies the model parameters, training rounds, and P2P network settings.

Supported Models
================

The framework is model-agnostic, allowing for the integration of various neural network models for anomaly detection. The default implementation includes a dense neural network specified in `models/net.py`, which can be easily replaced or extended based on the application requirements.

Federated Learning Process
=========================

1. **Initialization**: The global model is initialized, and configuration parameters are distributed to all participating nodes.
2. **Local Training**: Each node trains the model locally with its dataset.
3. **Model Update Sharing**: The nodes serialize their model updates and share them with the aggregator through a P2P network.
4. **Federated Averaging**: The aggregator deserializes the received updates, performs federated averaging, and updates the global model.
5. **Global Model Distribution**: The updated global model is then distributed back to the nodes for the next training round.

Repository Structure
====================

The **anomaly-dfl** project is structured as follows:

- `anomaly-dfl/main.py`: Entry point for the federated learning process.
- `anomaly-dfl/models/`: Contains the neural network models.
- `anomaly-dfl/network/`: Implements the P2P communication.
- `anomaly-dfl/utils/`: Provides utility functions for data loading and weight manipulation.
- `data/`: Hosts the dataset used for training.
- `tests/`: Contains unit tests for various components of the project.
- `docs/`: Documentation for the project.

Generating Documentation
========================

Documentation can be generated using Sphinx. Navigate to the `docs/` directory and run:

.. code-block:: bash

    make html

This will generate HTML documentation in the `docs/_build/html/` directory.

Versioning
==========

The current version of the project is maintained in the `VERSION` file. Ensure to update this file as the project evolves.

License
=======

**anomaly-dfl** is licensed under the MIT License. See the `LICENSE` file for more details.

Contributing
============

Contributions to **anomaly-dfl** are welcome! Please refer to the `CONTRIBUTING.md` file for guidelines on how to contribute to this project.

About the Author
================

Developed by Anastasios Kaltakis, **anomaly-dfl** reflects a dedication to advancing the field of federated learning with a focus on privacy-preserving techniques. With extensive experience in machine learning and software development, Anastasios has committed to creating a framework that empowers users to collaborate on machine learning tasks while maintaining the privacy of their data.