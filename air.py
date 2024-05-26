import os
import sys
import requests
import pymysql
import json
import pandas as pd
from sqlalchemy import create_engine

def air_API(servicekey): #servicekey에 따라서 air_API 호출할것
    url = "http://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getCtprvnRltmMesureDnsty"
    params = {'serviceKey':servicekey, #받아쓸거니까
          'returnType':'json',
          'numOfRows':'100',
          'pageNo':'1',
          'sidoName':'서울'}
    response = requests.get(url,params=params)
    print(response.status_code)
    if response.status_code!=200:
        print('API호출오류!')
        sys.exit(1)
    result = response.json()
    df = pd.DataFrame(result['response']['body']['items'])
    col = ['dataTime', 'sidoName', 'stationName', 'so2Value', 'coValue', 'o3Value','no2Value','pm10Value']
    df = df[col]
    return df 

def DB_insert(df, user, password, host, port, db, table_name):
    conn = 'mysql+pymysql://{}:{}@{}:{}/{}'.format(user,password,host,port,db)
    engine = create_engine(conn)
    df.to_sql(name=table_name, con=engine, if_exists='append', index=False)

def main(): #모듈 불러올때마다 실행할 필요 없으니까 함수로 설정
    my_df = air_API('SrBP4MJ4PYFoYhIxaHfCMHSC1yBK4gLjLleGwxw2QL0MA/Etyjf8JiX/2fhz5/sL4xrzckRXXjWNnS4agSqm5A==')
    user = 'lim'
    db = 'limdb'
    password = 'Asd2004'
    host = 'host.docker.internal' 
    port = 3307
    DB_insert(my_df, user, password, host, port, db, 'air')

#파이썬 실행되면 아무 결과 X (그냥 실행하고 끝남)
if __name__ == "__main__": #모듈로 불러오면 main함수 실행 x. 그렇지만 기능하는 애들은 할 수 있음
    main() 
