# Repository for 102 flowers classification using modified convolutional neural network
This repository contains the code, datasets, and models required for the paper.

## Requirements

- The code is implemented using **Python 3.11.9**.
- A list of required packages can be found in the `requirements.txt` file.

## How to Run

1. Navigate to the `code` directory.
2. Run the script `m-CNN_persian.py`.

## Data and Models

- The dataset and saved models are available in the `data` directory.

## Code Descriptions

The `code` directory contains several Python scripts for different classification models. Below is a description of each script:

- **TextOnly_BLSTM.py**: Implements text-only classification using an *Attention-Based Bidirectional Long Short-Term Memory Network (BLSTM)* model.

- **TextOnly_CNN.py**: Implements text-only classification using a *Convolutional Neural Network (CNN)* model.

- **TextOnly_LSTM.py**: Implements text-only classification using a *Long Short-Term Memory Networks (LSTM)* model.

- **m-CNN_Persian_ENetB2.py**: Implements classification using both text and image features. Image features are extracted using the *EfficientNet-b2* model, combined with text data for classification.

- **m-CNN_imageOnly.py**: Implements image-only classification.

- **m-CNN_persian.py**: Implements classification using both text and image features. The image features are extracted using the *VGG16* model, combined with text data for classification.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
