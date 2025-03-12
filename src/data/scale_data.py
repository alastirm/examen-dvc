import pandas as pd
import numpy as np
from pathlib import Path
import click
import logging
from sklearn.model_selection import train_test_split
import os

# scaling
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

@click.command()
@click.argument('input_filepath', type=click.Path(exists=False), required=0)
@click.argument('output_filepath', type=click.Path(exists=False), required=0)
def main(input_filepath, output_filepath):
    """ Runs data scaling to scale data (X_test, X_train) from (../../data/processed_data) into
        2 scaled datasets  (X_test_scaled, X_train_scaled) saved in ../../data/processed_data.
    """
    logger = logging.getLogger(__name__)
    logger.info('scaling training data set from processing data')

    input_filepath = click.prompt('Enter the file path for the input training data (X_test, X_train)', type=click.Path(exists=True))
    output_filepath = click.prompt('Enter the file path for the output scaled data (e.g., data/processed_data', type=click.Path())
    scale_data(input_filepath, output_filepath)

def scale_data(input_filepath, output_filepath):
    input_filepath_Xtrain = f"{input_filepath}/X_train.csv"
    input_filepath_Xtest = f"{input_filepath}/X_test.csv"
    X_train = import_dataset(input_filepath_Xtrain)
    X_test = import_dataset(input_filepath_Xtest)
    X_train = X_train.set_index("date")
    X_test = X_test.set_index("date")

    # On fit sur Xtrain complet
    scaler = MinMaxScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns = X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns = X_test.columns)

    # Create folder if necessary
    create_folder_if_necessary(output_filepath)

    # Save dataframes to their respective output file paths
    save_dataframes(X_train_scaled, X_test_scaled, output_filepath)


def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

def create_folder_if_necessary(output_folderpath):
    # Create folder if necessary
    if not os.path.exists(output_folderpath):
        os.makedirs(output_folderpath)
        
def save_dataframes(X_train_scaled, X_test_scaled, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train_scaled, X_test_scaled], ['X_train_scaled', 'X_test_scaled']):
        if os.path.exists(output_folderpath):
            output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
            file.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()