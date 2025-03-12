import pandas as pd
import numpy as np
from pathlib import Path
import click
import logging
from sklearn.model_selection import train_test_split
import os

@click.command()
@click.argument('input_filepath', type=click.Path(exists=False), required=0)
@click.argument('output_filepath', type=click.Path(exists=False), required=0)
def main(input_filepath, output_filepath):
    """ Runs data split to turn raw data from (../../data/raw_data) into
        4 datasets (X_test, X_train, y_test, y_train) saved in ../../data/processed_data.
    """
    logger = logging.getLogger(__name__)
    logger.info('making training data set from raw data')

    input_filepath = click.prompt('Enter the file path for the input data', type=click.Path(exists=True))
    input_filepath_data = f"{input_filepath}/raw.csv"
    output_filepath = click.prompt('Enter the file path for the output processed data (e.g., data/processed_data', type=click.Path())
    process_data(input_filepath_data, output_filepath)


def process_data(input_filepath_data, output_filepath):
    df_raw = import_dataset(input_filepath_data, sep=",")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df_raw)

    # Create folder if necessary
    create_folder_if_necessary(output_filepath)

    # Save dataframes to their respective output file paths
    save_dataframes(X_train, X_test, y_train, y_test, output_filepath)


def import_dataset(file_path, **kwargs):
    return pd.read_csv(file_path, **kwargs)

def split_data(df):
    # Split data into training and testing sets
    target = df['silica_concentrate']
    feats = df.drop(['silica_concentrate'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.3, random_state=1234)
    return X_train, X_test, y_train, y_test


def create_folder_if_necessary(output_folderpath):
    # Create folder if necessary
    if not os.path.exists(output_folderpath):
        os.makedirs(output_folderpath)
        
def save_dataframes(X_train, X_test, y_train, y_test, output_folderpath):
    # Save dataframes to their respective output file paths
    for file, filename in zip([X_train, X_test, y_train, y_test], ['X_train', 'X_test', 'y_train', 'y_test']):
        if os.path.exists(output_folderpath):
            output_filepath = os.path.join(output_folderpath, f'{filename}.csv')
            file.to_csv(output_filepath, index=False)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()