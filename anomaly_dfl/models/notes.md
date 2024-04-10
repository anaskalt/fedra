### Models Directory Structure

This directory houses various neural network models tailored for different machine learning tasks, including both classification and regression. The naming convention for model files is designed to offer a clear understanding of each model's characteristics at a glance. Each filename follows the structure:

```console
{details}{model_type}{task}.py
```

- `{details}`: Describes the model's specific features or its complexity level (e.g., `simple`, `complex`, `dropout`). This part of the name offers a hint about the model's architecture or special attributes that set it apart from others.

- `{model_type}`: Indicates the type of neural network (e.g., `nn` for neural network, `cnn` for convolutional neural network, `rnn` for recurrent neural network). This segment provides a straightforward classification of the model's foundational architecture.

- `{task}`: Specifies the primary machine learning task the model is designed for (`classification` or `regression`). This ensures users can quickly identify the model's intended application domain.

For example, a file named `simple_nn_classification.py` would contain a straightforward neural network model intended for classification tasks. This naming strategy helps maintain order within the directory, making it easier for developers and researchers to find or add new models as per their requirements.
