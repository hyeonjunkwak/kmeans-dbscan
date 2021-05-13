# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 00:49:20 2021

@author: user
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
from shapely.geometry import MultiPolygon, JOIN_STYLE
import itertools
from fiona.crs import from_string
from pyproj import CRS
import time
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn import datasets
import matplotlib.pyplot as plt
%matplotlib inline 
import warnings
warnings.filterwarnings(action='ignore') 
import folium
import os

epsg4326 = from_string("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs") # from_string은 좌표 지정할 때 쓰는 코드
epsg5174 = from_string("+proj=tmerc +lat_0=38 +lon_0=127.0028902777778 +k=1 +x_0=200000 +y_0=500000 +ellps=bessel +units=m +no_defs +towgs84=-115.80,474.99,674.11,1.16,-2.31,-1.63,6.43")
#epsg5179 = from_string("+proj=tmerc +lat_0=38 +lon_0=127.5 +k=0.9996 +x_0=1000000 +y_0=2000000 +ellps=GRS80 +units=m +no_defs")
#epsg5181 = from_string("+proj=tmerc +lat_0=38 +lon_0=127 +k=1 +x_0=200000 +y_0=500000 +ellps=GRS80 +units=m +no_defs")
epsg5181_qgis = from_string("+proj=tmerc +lat_0=38 +lon_0=127 +k=1 +x_0=200000 +y_0=500000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs") # qgis 좌표, 보정좌표값 존재
#epsg5186 = from_string("+proj=tmerc +lat_0=38 +lon_0=127 +k=1 +x_0=200000 +y_0=600000 +ellps=GRS80 +units=m +no_defs")
epsg2097 = from_string("+proj=tmerc +lat_0=38 +lon_0=127 +k=1 +x_0=200000 +y_0=500000 +ellps=bessel +units=m +no_defs +towgs84=-115.80,474.99,674.11,1.16,-2.31,-1.63,6.43")
cc=CRS({'init':'epsg:4326', 'no_defs':True})


#%%

# 단지형 아파트csv 불러오기

danji_apt=pd.read_csv(r'D:\부동산 빅데이터 분석 스터디\아파트 찾기 프로젝트 수정\서울 아파트.csv', encoding='cp949', sep=',')
del danji_apt['Unnamed: 0']
del danji_apt['geometry']
danji_apt_sexy=danji_apt.copy() # 실거래가 조회할때 쓰기 위함
danji_apt.rename(columns={'사용연수' : '경과연수', 'k-전체세대수' : '세대수', 'k-아파트명' : '건물 이름'}, inplace=True)
danji_apt=danji_apt[(danji_apt['세대수']>=200) | danji_apt['건물 이름'].str.contains('자이|힐스테이트|래미안|푸르지오|e편한세상|센트럴 아이파크|센트럴아이파크|롯데캐슬')] # 적어도 200세대 이상인 아파트(단지형 아파트 파일인데 은근 나홀로 아파트가 많음), 대충 2700개정도 날라감

danji_apt['geometry'] = danji_apt.apply(lambda row: Point(row.좌표X, row.좌표Y), axis=1)

danji_apt_geo=gpd.GeoDataFrame(danji_apt, geometry='geometry', crs=cc)

danji_apt_geo=danji_apt_geo.to_crs(epsg5181_qgis)

danji_apt_geo=danji_apt_geo.dropna(subset=['건물 이름']) 

#%%

# 먼저 생활하면서 필수적인 편의점 유무부터 거르고 가자. 

sangga_upso=pd.read_csv(r'D:\부동산 빅데이터 분석 스터디\아파트 찾기 프로젝트 수정\소상공인시장진흥공단_상가(상권)정보_서울_202012.csv', encoding='utf-8', sep='|')
sangga_upso=sangga_upso.loc[sangga_upso['상권업종소분류명'].str.contains('편의점')]
sangga_upso=sangga_upso.loc[sangga_upso['상호명'].str.contains('씨유|CU|Cu|cu|지에스|GS25|Gs25|gs25|GS|Gs|gs|세븐일레븐|코리아세븐|위드미|비지에프리테일|미니스톱|훼미리|비지에프|Emart|emart|이마트')]
sangga_upso=sangga_upso.rename(columns={'경도' : 'x', '위도' : 'y'})
sangga_upso['level'] = pd.qcut(sangga_upso.x, 10, labels=False)
sangga_upso['geometry']=sangga_upso.apply(lambda row : Point(row.x, row.y), axis=1)
cc3=CRS({'init':'epsg:4326', 'no_defs':True})
sangga_upso_geo=gpd.GeoDataFrame(sangga_upso, geometry='geometry', crs=cc3)
sangga_upso_geo=sangga_upso_geo.to_crs(epsg5181_qgis)

a=danji_apt_geo[danji_apt_geo['건물 이름']=='1']

for i in tqdm(range(10), desc="편의점 작업중...") :
    sangga_upso_geo1=sangga_upso_geo.loc[sangga_upso_geo['level']==i]

    sangga_upso_geo_buffer1=gpd.GeoDataFrame(sangga_upso_geo1.buffer(500))
    sangga_upso_geo_buffer1.columns=['geometry']
    sangga_upso_geo_buffer1=gpd.GeoDataFrame(sangga_upso_geo_buffer1, geometry='geometry')
    sangga_upso_geo_buffer1['new_column'] = 0
    sangga_upso_geo_buffer1_merge = sangga_upso_geo_buffer1.dissolve(by='new_column')

    danji_apt_convi=danji_apt_geo.loc[danji_apt_geo.geometry.within(sangga_upso_geo_buffer1_merge.geometry.unary_union)] # unary_union multipolygon의 합집합을 반환.
    
    a=pd.concat([a, danji_apt_convi]) 

b=a.drop_duplicates(subset='geometry') # 500m 이내에 편의점에 전혀 없는 아파트 26개가 걸러짐

danji_apt_geo=b

#%%

# 해당 아파트로부터 반경 450m 이내에 아파트 세대수 합계가 1500세대 이상인 아파트만 추출. 

# for문으로 각 아파트 buffer(450)를 그리고 그 원안에 속하는 아파트만 세대수 더하기.
count_ori=len(danji_apt_geo)

danji_apt_geo_sedae=danji_apt_geo.copy()
danji_apt_geo_sedae=danji_apt_geo_sedae.drop_duplicates('건물 이름')

count=len(danji_apt_geo_sedae)

danji_apt_geo_sedae=danji_apt_geo_sedae.reset_index()
del danji_apt_geo_sedae['index']

danji_apt_geo_sedae['450m 반경 내 아파트 세대수(해당 아파트 포함)']=''

for i in tqdm(range(count), desc='450m 반경 내 아파트 세대수 세는중...') :
    buf=danji_apt_geo_sedae.loc[i, 'geometry'].buffer(450)
    danji_apt_inter=danji_apt_geo_sedae.loc[danji_apt_geo_sedae.geometry.intersects(buf)]
    sedae_sum=np.sum(danji_apt_inter['세대수'])
    danji_apt_geo_sedae.loc[i, '450m 반경 내 아파트 세대수(해당 아파트 포함)']=sedae_sum # buf에 intersects하면 해당 아파트 자신은 무조건 포함되므로 따로 더할 필요 x

# 1000세대 이상으로 하면 200개, 1500세대 이상은 350개, 2000세대 이상은 500개 아파트 걸러짐    
danji_apt_geo_sedae=danji_apt_geo_sedae[danji_apt_geo_sedae['450m 반경 내 아파트 세대수(해당 아파트 포함)']>=1500]

danji_apt_geo=pd.merge(danji_apt_geo, danji_apt_geo_sedae[['건물 이름', '450m 반경 내 아파트 세대수(해당 아파트 포함)']], on='건물 이름', how='inner')

danji_apt_machine=danji_apt_geo.copy() # 맨 아래에서 kmeans에 쓸 데이터프레임 만들기

#%%

apart_input=input('검색기 결과 중 주변 아파트를 조회해보고 싶은 아파트 이름을 입력해주세요\n')

danji_apt_dbscan=danji_apt_machine.copy()
count=len(danji_apt_dbscan)
danji_apt_dbscan['x_meter']=''
danji_apt_dbscan['y_meter']=''
danji_apt_dbscan= danji_apt_dbscan.reset_index()
del danji_apt_dbscan['index']

for i in range(count) :
    danji_apt_dbscan.loc[i, 'x_meter']=danji_apt_dbscan['geometry'][i].coords.xy[0][0]
    danji_apt_dbscan.loc[i, 'y_meter']=danji_apt_dbscan['geometry'][i].coords.xy[1][0]

dbscan_df=danji_apt_dbscan[['x_meter', 'y_meter']]
dbscan_df=dbscan_df.rename(columns={'x_meter' : 'x', 'y_meter' : 'y'})

dbscan=DBSCAN(min_samples=2, eps=300) # 350m 이내면 하나의 군집으로 묶음
clusters=dbscan.fit_predict(dbscan_df) # -1은 잡음(군집되지 않은 점)

print(f'category : {clusters}')

plt.scatter(dbscan_df['x'],dbscan_df['y'], c=clusters, marker='o',s=10)

danji_apt_dbscan['category']=clusters

danji_apt_dbscan2=danji_apt_dbscan[['kapt도로명주소', '건물 이름', 'category']]

danji_apt_dbscan_noise=danji_apt_dbscan[danji_apt_dbscan['category']==-1]

danji_apt_dbscan3=danji_apt_dbscan[danji_apt_dbscan['category']!=-1]

# 입력한 아파트 찾기

category_num=danji_apt_dbscan3[danji_apt_dbscan3['건물 이름']==apart_input]['category'].values[0]

danji_apt_category=danji_apt_dbscan3[danji_apt_dbscan3['category']==category_num]

danji_apt_category=danji_apt_category.drop_duplicates('건물 이름')

lat = danji_apt_category['y'].mean()
long = danji_apt_category['x'].mean()

m = folium.Map([lat,long],zoom_start=16)

for i in danji_apt_category.index:
    sub_lat =  danji_apt_category.loc[i,'y']
    sub_long = danji_apt_category.loc[i,'x']
    
    title = danji_apt_category.loc[i,'건물 이름']
  
    color = 'blue'
    if danji_apt_category.loc[i, '건물 이름'] == apart_input:
        color = 'red'
    iframe=folium.IFrame(danji_apt_category.loc[i,'건물 이름'], width=450, height=60)
    popup=folium.Popup(iframe)
    folium.Marker([sub_lat,sub_long], popup=popup, icon=folium.Icon(icon='home', color=color)).add_to(m)

m.save('dnji_apt_DBSCAN.html')
