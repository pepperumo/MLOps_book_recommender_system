import os
import logging
import shutil
import kagglehub
from check_structure import check_existing_file, check_existing_folder


def import_raw_data(raw_data_relative_path, dataset_name):
    '''Import dataset from Kaggle using kagglehub into raw_data_relative_path'''
    if check_existing_folder(raw_data_relative_path):
        os.makedirs(raw_data_relative_path)
    
    # Download the dataset using kagglehub
    print(f'Downloading dataset: {dataset_name}')
    dataset_path = kagglehub.dataset_download(dataset_name)
    print(f'Dataset downloaded to: {dataset_path}')
    
    # Copy all files from the downloaded path to the raw data directory
    for filename in os.listdir(dataset_path):
        source_file = os.path.join(dataset_path, filename)
        destination_file = os.path.join(raw_data_relative_path, filename)
        
        if os.path.isfile(source_file) and check_existing_file(destination_file):
            print(f'Copying {filename} to {destination_file}')
            shutil.copy2(source_file, destination_file)
    
    print(f'All files copied to {raw_data_relative_path}')
                
def main(raw_data_relative_path="./data/raw", 
        dataset_name="zygmunt/goodbooks-10k"):
    """ Download data from Kaggle using kagglehub into ./data/raw
    """
    import_raw_data(raw_data_relative_path, dataset_name)
    logger = logging.getLogger(__name__)
    logger.info('making raw data set')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    main()
