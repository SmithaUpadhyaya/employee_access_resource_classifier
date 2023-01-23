from src.models.feature_eng.CountVectorizerEncoding import CountVectorizerEncoding
import utils.write_utils as hlpwrite
import utils.read_utils as hlpread
import sys
import os

#Load Cleaned data 
clean_train_data = os.path.join(hlpread.read_yaml_key('data_source.data_folders'),
                                hlpread.read_yaml_key('data_source.prepared.folder'),
                                hlpread.read_yaml_key('data_source.prepared.clean_train'),
                                )
db_train = hlpread.read_from_parquet(clean_train_data)

params = hlpread.read_yaml_key('featurize.count_vector')
target_col = 'ACTION'

count_obj = CountVectorizerEncoding(targetcol = target_col, combine_cols = True)
new_dataset = count_obj.fit_transform(db_train)

save_file_path = os.path.join(
                                hlpread.read_yaml_key('data_source.data_folders'),
                                params['output']['folder'],
                                params['output']['filename'],                                  
                                )

hlpwrite.save_to_parquet(new_dataset, save_file_path, True)

"""
if len(sys.argv) != 1:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython count_feature_vectorizer.py count_vector\n")
    #sys.stderr.write("\tpython featurization.py [tfidf|count_vector|TE|KTE|FE|KFE]\n")
    sys.exit(1)


feature_code = sys.argv[1].lower()

#if feature_code not in ['tfidf', 'count_vector','TE','KTE','FE','KFE']:
if feature_code != 'count_vector':
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython count_feature_vectorizer.py count_vector\n")
    #sys.stderr.write("\tpython featurization.py [tfidf|count_vector|TE|KTE|FE|KFE]\n")
    sys.exit(1)

#LOAD CLEANED DATA 

target_col = 'ACTION'
col_use = [x for x in db_train.columns if not x in ['ROLE_TITLE', 'MGR_ID']]
db_train = db_train[col_use]

db_train = combine_cols(db_train, col_use, target_col)

if feature_code == 'count_vector':

    params = hlpread.read_yaml_key('featurize.count_vector')
    new_dataset = get_col_interactions_svd(db_train, feature_code, target_col, params)

    #SAVE THE FILE USING hlpread

"""


