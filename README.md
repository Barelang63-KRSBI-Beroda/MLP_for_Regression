# **MLP_for_Regression**
This repository provides tools to train and make predictions using a Multi-Layer Perceptron (MLP) for regression tasks, implemented with PyTorch.

As an example, the dataset used here is from Kaggle:

[🔗 Student Performance Dataset](https://www.kaggle.com/datasets/larsen0966/student-performance-data-set?resource=download)

Before training the model, make sure your dataset has gone through appropriate feature engineering, such as:
* Converting categorical data into numeric (e.g., one-hot or label encoding)
* Handling outliers
* Any other steps necessary to prepare the data

## **📦 Installation**
Install the required dependencies:
```
pip3 install -r requirements.txt
```

## **🚀 Training the Model**
To train the model, use the train.py script.  
This script requires a configuration file `config.yaml`.
An example can be found in the `config/` directory:

```
data:
  path: "dataset/student_performance.csv"
  target_column: "Performance Index"
  test_size: 0.2

training:
  num_epochs: 500
  batch_size: 16
  learning_rate: 0.001
  patience: 50

model:
  hidden_layers: [32, 16, 8]
  activation: "tanh"
```

### 🔧 Configuration Notes:
* Adjust `data.path` to the location of your dataset file.
* Set `target_column` to the name of the column you want to predict (e.g., `"Performance Index"`).
* You can modify `hidden_layers` and other hyperparameters as needed.
* The `activation` parameter supports only three activation functions by default: `sigmoid`, `relu`, and `tanh`. If you wish to use a different activation function, make sure it is supported by PyTorch and modify the implementation manually in `utils/model.py`.

### ▶️ Running the Training Script
Once your configuration file is ready, you can start training the model with the following command:
```
python3 train.py --config config/config.yaml
```
By default, the script will automatically create an output folder in the `results/` directory with a name like `model_1`.  

You can also manually specify the output directory:
```
python3 train.py --config config/config.yaml --output results/student_performance_model
```
When training is running, you’ll see output similar to the following: 
```
Dataset loaded from dataset/student_performance.csv  
Dataset shape: (9984, 6)  
   Hours Studied  Previous Scores  ...  Sample Question Papers Practiced  Performance Index    
0              7               99  ...                                 1               91.0  
1              4               82  ...                                 2               65.0  
2              8               51  ...                                 2               45.0  
3              5               52  ...                                 2               36.0  
4              7               75  ...                                 5               66.0  
5              3               78  ...                                 6               61.0  

[6 rows x 6 columns]    
Data split: Train=7987 samples, Test=1997 samples  
Model initialized with architecture: [32, 16, 8]   
Using device: cuda  
Starting training for 500 epochs with batch size 16  
Epoch [1/500], Train Loss: 0.084344, Test Loss: 0.024285 (Best)        
Epoch [2/500], Train Loss: 0.020212, Test Loss: 0.017671 (Best)   
Epoch [3/500], Train Loss: 0.016599, Test Loss: 0.015175 (Best)   
Epoch [4/500], Train Loss: 0.014596, Test Loss: 0.013770 (Best)   
Epoch [5/500], Train Loss: 0.013353, Test Loss: 0.013166 (Best)   
....
```

## **🧪 Predicting the Model**
Once training is complete and the model has been saved, you can perform inference and evaluate the model using the `predict.py` script.

### ▶️ Example command:
```
python3 predict.py \
  --model results/model_1/best_model/model_full.pth \
  --scaler-x results/model_1/scaler_X.pkl \
  --scaler-y results/model_1/scaler_y.pkl \
  --input dataset/predict_student_performance.csv \
  --target-col "Performance Index"
```
Make sure that:

* `--model` points to the trained model file (`.pth`)

* -`-scaler-x` and `--scaler-y` point to the respective saved scalers

* `--input` is the CSV file containing input features and the true target values

* `--target-col` matches the name of the column containing the ground-truth targets (e.g., `"Performance Index"` — make sure to wrap names with spaces in quotes)

### 📘 Check available arguments:

To view all available options and usage, run:
```
python3 predict.py -h
```
You will see output like:
```
usage: predict.py [-h] --model MODEL --scaler-x SCALER_X --scaler-y SCALER_Y --input INPUT
                  [--output OUTPUT] [--input-size INPUT_SIZE] [--hidden-layers HIDDEN_LAYERS]
                  [--target-col TARGET_COL]

Make predictions with a trained regression model.

options:
  -h, --help            show this help message and exit
  --model MODEL         Path to the saved model file
  --scaler-x SCALER_X   Path to the saved feature scaler
  --scaler-y SCALER_Y   Path to the saved target scaler
  --input INPUT         Path to input CSV file with features and expected target
  --output OUTPUT       Base path to save predictions (e.g., predictions.xlsx)
  --input-size INPUT_SIZE
                        Input size (needed only if loading state dict)
  --hidden-layers HIDDEN_LAYERS
                        Hidden layers as comma-separated values (needed only if loading state dict)
  --target-col TARGET_COL
                        Name of the column in input CSV with expected target values
```
### 📤 Prediction Output
The script will generate an Excel file (`.xlsx`) containing:
* **Predicted Value**: Output from the model

* **Actual Value**: True label from your dataset

* **Error**: Absolute error between predicted and actual

A second sheet in the Excel file provides evaluation metrics:

* **MSE** (Mean Squared Error)

* **RMSE** (Root Mean Squared Error)

* **MAE** (Mean Absolute Error)

* **R²** (R-squared Score)

By default, the output will be saved as `predictions.xlsx` in the current directory, unless otherwise specified using the `--output` argument.

