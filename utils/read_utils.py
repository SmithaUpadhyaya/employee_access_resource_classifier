import utils.paths as pth
import pandas as pd
import numpy as np
import yaml
import dill

params_file_path = pth.get_configuration_path()

def read_csv(file_path, **kargs):

    """
    read csv file
    file_path: str relative path of the file to read.
    return: dataframe pandas/cudf.
    """   
     
    if len(kargs.keys()) == 3:
        return pd.read_csv(file_path, 
                        converters = kargs['converters'],  
                        usecols = kargs['usecols'],
                        dtype = kargs['dtype'],
                        )

    else:
        return pd.read_csv(file_path)

def read_from_parquet(file_path):

    """
    read parquet file.    
    file_path: str relative path of the file to read.
    return: dataframe pandas/cudf.
    """
    
    return pd.read_parquet(file_path)
    
def read_yaml_file(file_path = None):
    
    """
    read yaml file.
    file_path: str relative path of the yaml file to read.
    return: dictonary    
    """    

    #log.write_log(f'Read yaml file from path {file_path} ...', log.logging.DEBUG)

    #if not os.path.exists(file_path):
    #    raise Exception(f"The file: {file_path} does not exists.")
        
    if file_path == None:
        file_path = params_file_path

    with open(file_path, "rb") as yaml_file:
        return yaml.safe_load(yaml_file)

def read_yaml_key(key, subkey = None, file_path = None):

    """
    read specify key from the yaml config file
    file_path: str relative path of the yaml file to read.
    key: str key to read from the yaml file
    subkey: str subkey to read from the key
    """
    if file_path == None:
        file_path = params_file_path

    config = read_yaml_file(file_path)
    value = config[key]

    if subkey != None:
        value = value[subkey]

    return value    

def read_compressed_numpy_array_data(file_path):

    """
    load compressed numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """   

    #if not os.path.exists(file_path):
    #    raise Exception(f"The file: {file_path} does not exists.")

    return np.load(file_path)['arr_0']    

def read_object(file_path: str, ) -> object:
       
    #if not os.path.exists(file_path):
    #    raise Exception(f"The file: {file_path} does not exists.")

    with open(file_path, "rb") as file_obj:
        return dill.load(file_obj)
    
    