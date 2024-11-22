import os
import boto3
import zipfile
import argparse
import sys
sys.path.append("..")
from utils.object_information import ObjectInformation
from utils import utilities

def download_object(bucket_name, obj_id, name, local_base_path, keep_prev=True, download_raw_data=False, extract_raw_data=False, collected_data_type='all'):
    """
    Downloads an object from AWS. Can optionally apply filters to only download some subset of the data. 
    Parameters: 
        bucket_name (str): Name of the AWS bucket
        obj_id (str): Unique ID of object to download
        name (str): Name of object to download
        local_base_bath (str): Path to store dataset in
        keep_prev (bool): If True, skip files that have been previously downloaded. If False, overwrite existing files. Default: True
        download_raw_data (bool): If True, download raw data zip file. If False, skip raw data. Default: False
        extract_raw_data (bool): If True (and raw data is downloaded), extract the raw data. Default: False
        collected_data_type (str): 
                Options: 'all': download both simulation and real-world data.
                         'robot_collected': download real-world data.
                         'simulation': download simulation data        
    Returns: None
    """
    assert collected_data_type in ['all', 'robot_collected', 'simulation'], 'Please set collected_data_type to all, robot_collected, or simulation'

    folder_name = f"{obj_id}_{name}" 

    s3 = boto3.client('s3')
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(bucket_name)

    for object in bucket.objects.all():
        key = object.key
        if folder_name not in key: continue
        local_path = os.path.join(local_base_path, key)
    
        # Skip raw data if applicable
        if not download_raw_data and 'unprocessed' in key: continue
        
        # Check if data matches requested data type
        if collected_data_type != 'all' and collected_data_type not in key: continue

        # Don't overwrite file unless requested
        if not keep_prev or (keep_prev and not os.path.exists(local_path)): 
            # Ensure the local directory exists before downloading
            local_dir = os.path.dirname(local_path)
            os.makedirs(local_dir, exist_ok=True)

            # Download the file
            s3.download_file(bucket_name, key, local_path)
            print(f'Downloaded File: {key} to {local_path}')
        
        # Extract raw data zip file if applicable
        if extract_raw_data and 'unprocessed' in key:
            local_path_folder = local_path.rpartition(', ')[0]
            with zipfile.ZipFile(local_path, 'r') as zip_ref:
                zip_ref.extractall(local_path_folder)

    print(f'Finished downloading object: {folder_name}')

def _list_objects(bucket_name):
    """
    List all objects within a bucket
    Parameters:
        bucket_name (str): Name of bucket to list

    Returns:
        all_objects (list of str): List of all object names in bucket
    """
    s3 = boto3.client('s3')
    objects_list = s3.list_objects_v2(Bucket=bucket_name, Delimiter='/')

    all_objects = []
    for object in objects_list.get('CommonPrefixes', []):
        object_name = object['Prefix']
        all_objects.append(object_name)

    return all_objects

def get_objects(bucket_name, local_base_path, objs='all', keep_prev=True, download_raw_data=False, extract_raw_data=False, collected_data_type='all'):    
    """
    Downloads an object from AWS. Can optionally apply filters to only download some subset of the data. 
    Parameters: 
        bucket_name (str): Name of the AWS bucket
        local_base_bath (str): Path to store dataset in
        objs (str or list of str): Which objects to download. The options are:
                - 'all': download all objects in AWS
                - 'sample': download a small sample of data
                - List of object IDs as strings (e.g., ['001', '002']): Download objects specified by their ID
                - List of object names as strings (e.g., ['wrench', 'spatula']): 
        keep_prev (bool): See download_object documentation
        download_raw_data (bool): See download_object documentation
        extract_raw_data (bool): See download_object documentation
        collected_data_type (str): See download_object documentation
                                   
    Returns: None
    """
    # If downloading all objects, list them all
    if objs=='all':
        folders = _list_objects(bucket_name=bucket_name)
        objs = []
        for obj in folders:
            obj_id, _, name = obj.partition('_')#[[0,2]]
            objs.append([obj_id, name])
    elif objs == 'sample':
        objs = [('022', 'windex_bottle'), ('032', 'knife'), ('033', 'spatula'), ('038', 'padlock'), ('042', 'wrench')]
    else:
        obj_info = ObjectInformation()
        objs = obj_info.convert_partial_obj_list(objs)

    # Download all data
    for obj in objs:
        obj_id, name = obj[0], obj[1]
        download_object(bucket_name, obj_id, name, local_base_path, keep_prev=keep_prev, download_raw_data=download_raw_data, 
                        extract_raw_data=extract_raw_data, collected_data_type=collected_data_type)



if __name__ == '__main__':

    bucket_name = 'mmwave-dataset-3' # AWS bucket name
    local_base_path = f'{utilities.get_root_path()}/data' # relative path to data
    
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Choose between downloading the full dataset or a sample.")

    # Add argument for choosing dataset type
    parser.add_argument("--download_data",
                        default="sample",
                        help="Choose 'all' to download the entire dataset or 'sample' for a smaller portion.You can also use a list obj ids or names.")

    # Parse the arguments
    args = parser.parse_args()

    # Choose which objects to download. Either 'all', 'sample', a list of object ids, or a list of object names
    # Examples: objs=['001', '004', '007'] or objs=['wrench', 'spatula'] or objs='all' or objs='sample'
    objs = args.download_data# ['spatula'] #'sample'
    objs = 'all'#['knife']

    get_objects(bucket_name, local_base_path, objs, download_raw_data = False, collected_data_type='simulation')

    # TODO: Remove unzipping file.
    # TODO: Remove spatula hardcode, download raw data
    # TODO: Split into raw data radar vs camera