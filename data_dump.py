import pymongo
import pandas as pd
import json
from dotenv import load_dotenv
# Provide the mongodb localhost url to conpytjon ct python to mongodb.

from thyroid.config import mongo_client



data_file_path='/config/workspace/thyroid0387 (1).data'
Database_name='ThyroidF'
collection_name='Thyroid Project'


if __name__=='__main__':

    df=pd.read_csv(data_file_path)
    # df.to_csv("thyroid.csv",index=None)   #creating base file for validation
    
    print((df.shape))

#convert datafrsam to json

df.reset_index(drop=True,inplace=True)

json_record=list(json.loads(df.T.to_json()).values())
print(json_record[0])

mongo_client[Database_name][collection_name].insert_many(json_record)