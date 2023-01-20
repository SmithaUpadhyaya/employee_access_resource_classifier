import os
#import argparse
import logs.logger as log
import utils.read_utils as hlpread
import utils.write_utils as hlwrite

if __name__ == '__main__':    

    #args = argparse.ArgumentParser()
    #args.add_argument("--config", default = "config/config.yaml")
    #parsed_args = args.parse_args()

    #config = parsed_args.config

    #log.write_log(f'Read configuration from path: {parsed_args.config}', log.logging.INFO)
    log.write_log(f'Clean data started.', log.logging.DEBUG)

    train_data = os.path.join( 
                             hlpread.read_yaml_key('data_source.data_folders'),
                             hlpread.read_yaml_key('data_source.training_data_folder.folder'),
                             hlpread.read_yaml_key('data_source.training_data_folder.train'), 
                            )

    #train_data

    db_train = hlpread.read_csv(train_data)

    #print(db_train.shape)
    #db_train.head()

    #Records found while EDA with same description by difference in ACTION and MAG_ID
    db_train.drop(db_train[db_train.RESOURCE.isin([27797, 27831, 36629, 77955, 81502])][db_train.ACTION ==1].index, axis = 0, inplace = True)
    db_train.shape

    #Drop the duplicate records from db_train
    #print(f'Before droping duplicate records: {db_train.shape[0]}')
    db_train.drop_duplicates(subset = ['ACTION', 'RESOURCE', 
                                       'ROLE_ROLLUP_1', 'ROLE_ROLLUP_2', 'ROLE_DEPTNAME', 
                                       'ROLE_TITLE', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY','ROLE_CODE'],
                            inplace = True                         
                            )

    #db_train.shape
    #print(f'After droping duplicate records: {db_train.shape[0]}')

    log.write_log(f'Clean data completed.', log.logging.DEBUG)
    
    #Save
    save_path = os.path.join(hlpread.read_yaml_key('data_source.data_folders'),
                             hlpread.read_yaml_key('data_source.prepared.folder'),
                             hlpread.read_yaml_key('data_source.prepared.clean_train'),
                            )
    hlwrite.save_to_parquet(db_train, save_path)

    log.write_log(f'Saved clean data at {save_path}.', log.logging.DEBUG)

