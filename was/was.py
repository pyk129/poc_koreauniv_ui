# -*- coding: utf-8 -*-
# 라이브러리 import
import json
import pandas as pd
import numpy as np
from haversine import haversine, Unit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

#import torch
#from transformers import BertTokenizer
#from keras.preprocessing.sequence import pad_sequences
import joblib

from datetime import datetime
        
class Recommend1:
    
    path_아파트 = 'datas/apt_list_seoul/apt_data_노원구.json'
    path_댓글데이터 = 'datas/naver_comments/result_contents_네이버카페_노원구.json'
        
    path_초등학교 = 'datas/data_ele'
    path_중학교 = 'datas/data_mid'
    path_고등학교 = 'datas/data_high'

    path_학원 = 'datas/data_edu'
    path_마트 = 'datas/data_mart'
    path_백화점 = 'datas/data_deptment' 
    path_지하철 = 'datas/data_subway'
    
    path_병원 = 'datas/data_hosp'
    path_공원 = 'datas/data_park'
    
    path_실거래가 = 'datas/apt_realprice/dataApt실거래가_노원구_201701_201912.json'
    path_실거래가_2020 = 'datas/apt_realprice2/dataApt실거래가_노원구_2020.json'
     
    path_2ndModel = 'datas/20200709_recomment2nd.pkl'
    path_3ndModel = 'C:/Users/pyk12/Downloads/recommendModel3rd_model'
    
    def loadData(self):
        # 노원구 시설 데이터 로드 
        self.json_apt_datas = self.openFile(self.path_아파트)
# =============================================================================
#         self.df실거래가 = pd.DataFrame(self.openFile(self.path_실거래가))      
# =============================================================================
        df실거래가tmp = pd.DataFrame(self.openFile(self.path_실거래가)) 
        self.df실거래가_2020 = pd.DataFrame(self.openFile(self.path_실거래가_2020))      
        self.df실거래가_2020['매매가격지수'] = 0
        self.df실거래가_2020['통화금융지표'] = 0
        self.df실거래가_2020['기준금리'] = 0
        self.df실거래가_2020['매매가격지수y'] = 0
        
# =============================================================================
#         print("-----------------------------------------------")                
#         maxPrice = str(df실거래가tmp['거래금액(만원)'].astype(int).max())           
#         maxDf = df실거래가tmp[df실거래가tmp['거래금액(만원)'] >= maxPrice]
#         print(maxDf.head())
#         print("-----------------------------------------------")       
#         
# =============================================================================
# =============================================================================
#         self.df실거래가_2020['거래금액(만원)'] = self.df실거래가_2020['거래금액(만원)']
# =============================================================================
# =============================================================================
#         self.df실거래가_2020['본번'] = pd.to_numeric(self.df실거래가_2020['본번'])
#         self.df실거래가_2020['부번'] = pd.to_numeric(self.df실거래가_2020['부번'])
#         self.df실거래가_2020['전용면적(㎡)'] = pd.to_numeric(self.df실거래가_2020('전용면적(㎡)'])
#         self.df실거래가_2020['계약년월'] = pd.to_numeric(self.df실거래가_2020['계약년월'])
#         self.df실거래가_2020['계약일'] = pd.to_numeric(self.df실거래가_2020['계약일'])
# # =============================================================================
# #         self.df실거래가_2020['거래금액(만원)'] = pd.to_numeric(self.df실거래가_2020['거래금액(만원)'].str.replace(pat=',', repl='', regex=False))
# # =============================================================================
#         self.df실거래가_2020['층'] = pd.to_numeric(self.df실거래가_2020['층'])
#         self.df실거래가_2020['건축년도'] = pd.to_numeric(self.df실거래가_2020['건축년도'])
# 
# =============================================================================
        self.df실거래가 = pd.concat([df실거래가tmp, self.df실거래가_2020])

        
        
    def createDataFromJson(self, datas):
        ret = []
        try:
            for sc in datas:
                d = {}
                d['LAT_WGS'] = sc['LAT_WGS']
                d['LON_WGS'] = sc['LON_WGS']

                d['이름'] = sc['이름']
                d['구분'] = sc['구분']
                d['상세구분'] = sc['상세구분']
                ret.append(d)
        except:
            pass
        
        return ret
    
    def createQuery(self, select):
        active_limitkm = 0.5
        inactive_limitkm = 0.5
        
        sel = float(select)
        
        if sel > 0.0:
            return [sel, active_limitkm, 0]
        else:
            return [sel, inactive_limitkm, 0]
    
    
    def __init__(self):
        
# =============================================================================
#         self.device = torch.device('cpu') 
#         self.model = torch.load(self.path_3ndModel, map_location=self.device)x
#         self.model.eval()
# =============================================================================
        
# =============================================================================
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
# =============================================================================
        
        # 출력 
        self.clf_from_joblib = joblib.load(self.path_2ndModel) 

                
# =============================================================================
#         dfAptInfo = self.df실거래가[
#                 (self.df실거래가['도로명'] == '월계로45가길') &
#                 (self.df실거래가['도로명건물본번호코드'] == '94'.zfill(5)) &
#                 (self.df실거래가['거래금액'].astype(int) >= 300000000)]
#         print('** :', len(dfAptInfo))
# =============================================================================

        self.json_school_ele_datas = self.openFile(self.path_초등학교)
        self.json_school_mid_datas = self.openFile(self.path_중학교)
        self.json_school_high_datas = self.openFile(self.path_고등학교)

        self.json_edu_datas = self.openFile(self.path_학원)
        self.json_mart_datas = self.openFile(self.path_마트)
        self.json_deptstore_datas = self.openFile(self.path_백화점)
        
        self.json_subway_datas = self.openFile(self.path_지하철)
        
        self.json_hosp_datas = self.openFile(self.path_병원)
        self.json_park_datas = self.openFile(self.path_공원)
       
        # item 데이터 생성 
        itemSchool_ele = [] 
        itemSchool_mid = [] 
        itemSchool_high = [] 
        itemEdu = []

        itemMart = []
        itemDeptStore = []
        itemSubway = []
        
        itemHosp = []
        itemPark = []

        self.itemKeys = ['itemSchool_ele', 'itemSchool_mid', 'itemSchool_high', 
                         'itemEdu', 'itemMart','itemDeptStore', 
                         'itemSubway','itemHosp','itemPark'] 
        self.itemLenKeys = ['초등학교','중학교','고등학교',
                            '학원', '마트','백화점', 
                            '지하철' ,'병원', '공원'] 
        
        for sc in self.json_school_ele_datas:
            d = {}
            d['LAT_WGS'] = sc['위도']
            d['LON_WGS'] = sc['경도']

            d['이름'] = sc['학교명']
            d['구분'] = '학교'

            d['상세구분'] = sc['학교급구분']
            itemSchool_ele.append(d)

        for sc in self.json_school_mid_datas:
            d = {}
            d['LAT_WGS'] = sc['위도']
            d['LON_WGS'] = sc['경도']

            d['이름'] = sc['학교명']
            d['구분'] = '학교'

            d['상세구분'] = sc['학교급구분']
            itemSchool_mid.append(d)


        for sc in self.json_school_high_datas:
            d = {}
            d['LAT_WGS'] = sc['위도']
            d['LON_WGS'] = sc['경도']

            d['이름'] = sc['학교명']
            d['구분'] = '학교'

            d['상세구분'] = sc['학교급구분']
            itemSchool_high.append(d)    

       
        for sc in self.json_subway_datas:
            try:      
                d = {}
                d['LAT_WGS'] = sc['LAT_WGS']
                d['LON_WGS'] = sc['LON_WGS']

                d['이름'] = sc['이름']
                d['구분'] = '지하철'

                d['상세구분'] = sc['상세구분']
                itemSubway.append(d)
            except:
                pass    
        
        try: 
            for sc in self.json_mart_datas:
                d = {}

                d['LAT_WGS'] = sc['LAT_WGS']
                d['LON_WGS'] = sc['LON_WGS']

                d['이름'] = sc['이름']
                d['구분'] = '마트'

                d['상세구분'] = '마트'

                itemMart.append(d)
        except:
            pass
        
        try: 
            for sc in self.json_deptstore_datas:
                d = {}

                d['LAT_WGS'] = sc['LAT_WGS']
                d['LON_WGS'] = sc['LON_WGS']

                d['이름'] = sc['이름']
                d['구분'] = sc['구분']

                d['상세구분'] = sc['상세구분']

                itemDeptStore.append(d)
                
        except:
            pass
        
     
        itemEdu = self.createDataFromJson(self.json_edu_datas)
        itemHosp = self.createDataFromJson(self.json_hosp_datas)
        itemPark = self.createDataFromJson(self.json_park_datas)
        
        self.items = [itemSchool_ele, itemSchool_mid, itemSchool_high, itemEdu, itemMart, itemDeptStore, itemSubway, itemHosp, itemPark] 


             
    def setup(self, simirarity, gu, optSecond, sel1, sel2, sel3, sel4, sel5, sel6, sel7,sel8, sel9, myprice):
         # [선택조건Index, 제한거리(km)]
        self.simirarity = simirarity
        self.guName = gu
        self.path_아파트 = 'datas/apt_list_seoul/apt_data_'+gu+'.json'
        self.path_실거래가 = 'datas/apt_realprice/dataApt실거래가_'+gu+'_201701_201912.json'
        self.path_실거래가_2020 = 'datas/apt_realprice2/dataApt실거래가_'+gu+'_2020.json'
        
        self.optSecond = optSecond
        self.myprice = myprice
        
        query = []
        query.append(self.createQuery(sel1))
        query.append(self.createQuery(sel2))
        query.append(self.createQuery(sel3))
        query.append(self.createQuery(sel4))
        query.append(self.createQuery(sel5))
        query.append(self.createQuery(sel6))
        query.append(self.createQuery(sel7))
        query.append(self.createQuery(sel8))
        query.append(self.createQuery(sel9))
        
        self.select_query = query
        print('select query : ', self.select_query)
        
# =============================================================================
#         self.select_query = [
#                          [0.0, inactive_limitkm],
#                          [0.0, inactive_limitkm],
#                          [0.0, inactive_limitkm],
#                          [0.0, inactive_limitkm],
#                          [1.0, active_limitkm],
#                          [1.0, active_limitkm]
#                         ]
# =============================================================================
        
# =============================================================================
#         
#         path = "/Users/a60067648/Downloads/chromedriver"
#         
#         # chrome 드라이버 
#         options = webdriver.ChromeOptions()
#         options.add_argument('headless')
#         options.add_argument('window-size=1920x1080')
#         options.add_argument("disable-gpu")
#         # 혹은 options.add_argument("--disable-gpu")
#         
#         self.driver = webdriver.Chrome(path, chrome_options=options)
# =============================================================================
                
        self.loadData()
        
        
    # 입력 데이터 변환
    def convert_input_data(self, sentences):
    
        # BERT의 토크나이저로 문장을 토큰으로 분리
        tokenized_texts = [self.tokenizer.tokenize(sent) for sent in sentences]
    
        # 입력 토큰의 최대 시퀀스 길이
        MAX_LEN = 128
    
        # 토큰을 숫자 인덱스로 변환
        input_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
        
        # 문장을 MAX_LEN 길이에 맞게 자르고, 모자란 부분을 패딩 0으로 채움
        input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    
        # 어텐션 마스크 초기화
        attention_masks = []
    
        # 어텐션 마스크를 패딩이 아니면 1, 패딩이면 0으로 설정
        # 패딩 부분은 BERT 모델에서 어텐션을 수행하지 않아 속도 향상
        for seq in input_ids:
            seq_mask = [float(i>0) for i in seq]
            attention_masks.append(seq_mask)
    
        # 데이터를 파이토치의 텐서로 변환
        inputs = torch.tensor(input_ids)
        masks = torch.tensor(attention_masks)
    
        return inputs, masks
    
    # 문장 테스트
    def test_sentences(self,sentences):
    
        # 평가모드로 변경
        self.model.eval()
    
        # 문장을 입력 데이터로 변환
        inputs, masks = self.convert_input_data(sentences)
    
        # 데이터를 GPU에 넣음
        b_input_ids = inputs.to(self.device)
        b_input_mask = masks.to(self.device)
                
        # 그래디언트 계산 안함
        with torch.no_grad():     
            # Forward 수행
            outputs = self.model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
    
        # 로스 구함
        logits = outputs[0]
    
        # CPU로 데이터 이동
        logits = logits.detach().cpu().numpy()
    
        return logits    
        
    
    def run(self, multi_nb):
        
        queryTrueValueCnt = 0
        
        for d in self.select_query:
            if d[0] > 0:
                queryTrueValueCnt += 1

        print("선택된 편의시설 개수 : ", queryTrueValueCnt) 
    
           
        # 아파트 별 거리 구함 
        for i, apt_data in enumerate(self.json_apt_datas):
            
            # 1. 내가 선택한 것의 개수가 똑같게 나왔는지 확인     
            queryDataEqualCount = queryTrueValueCnt 
            # 2. 내가 선택한거 외에 추가적인 아이템 개수는 몇개인가?
            notSelItemCnt = 0
            
            totalkm = 0
            
            queryDataEqualCounts = []
            queryDataNotSelItemCnt = []
            queryDataTotalkm = []
            
            for idx, d in enumerate(self.select_query): 
             
                ret = []

                itemLimitKm = d[1]
                
                if itemLimitKm < 0:
                    continue
                
                ret, selQueryInRangeItemTotalkm = self.getInRangeSelectData(apt_data, self.items[idx], itemLimitKm)
                apt_data[self.itemKeys[idx]] = ret                
                
                # 만약 데이터가 없다면..
                resCnt = len(ret)
                
                if d[0] > 0:
                    # 선택한 쿼리 
                    if resCnt <= 0:
                        # 같지 않다. 
                        queryDataEqualCount -= 1
                    
                    totalkm += selQueryInRangeItemTotalkm
                    
                else:
                    # 선택하지 않은 쿼리
                    if resCnt > 0:
                        notSelItemCnt += 1
                    
                apt_data[self.itemLenKeys[idx]] = len(ret) + d[2]
                
                # 값 보정 
                if queryDataEqualCount < 0:
                    queryDataEqualCount = 0
                
                if d[0] > 0:
                    # 내가 선택한 데이터와 같은애의 개수
                    apt_data['queryDataEqualCount'] = int(queryDataEqualCount)
                    
                    # 내가 선택한 애의 전체 개수
                    apt_data['queryDataTotalCount'] = queryTrueValueCnt
                    # 내가 선택하지 않은 데이터의 개수 
                    apt_data['queryDataNotSelItemCnt'] = notSelItemCnt
                    # 내가 선택한 데이터의 전체 km 
                    apt_data['queryDataTotalkm'] = totalkm
                                       
# =============================================================================
#         queryDataTotalkm
#         queryDataTotalCount - 
# # =============================================================================
        
# =============================================================================
#                 print('*** : ', self.itemLenKeys[idx], ' ', len(ret))
# =============================================================================
              
# =============================================================================
#                 print('!!! : ', self.itemLenKeys[idx] , " data : " , len(ret))
# =============================================================================

        df = pd.DataFrame(self.json_apt_datas, columns = self.itemLenKeys)
    
        print(df.head())
        df = df.fillna(0)
        
        #################################
        # 유사도 측정 
        #################################
        df_matrix = df.values
        df_matrix_MinMax = df_matrix
# =============================================================================
#         min_max_scaler = MinMaxScaler()
#         df_matrix_MinMax = min_max_scaler.fit_transform(df_matrix)
# =============================================================================
        
        if self.simirarity == '1':
            print('*************************************************')
            print('*      유사도 비교 방법 : 코사인 유사도 비교       *')
            print('*************************************************')
            #################################
            # 코사인 유사도 
            #################################
            ret_simularity = df_matrix_MinMax
        elif self.simirarity == '2':
            #################################
            # Matrix Factrazation 
            ##################################
            print('*************************************************')
            print('       유사도 비교 방법   :  Matrix Factrazation 비교 ')
            print('*************************************************')
            factorizer = MatrixFactorization(df_matrix, k=200, learning_rate=0.001, reg_param=0.01, epochs=50, verbose=True)
#            factorizer = MatrixFactorization(df_matrix_MinMax, k=6, learning_rate=0.21, reg_param=0.01, epochs=200, verbose=True)
            factorizer.fit()
            print('*************************************************')
            print(' Matrix Factrazation 전 matrix ')
            print('*************************************************')
            print(' ')
            print(df_matrix)
            print(' ')  
            print('*************************************************')
            
            ret_simularity = factorizer.get_complete_matrix()
            print(' ')
            print(' ')
            print('*************************************************')
            print(' Matrix Factrazation 적용 matrix ')
            print('*************************************************')
            print(' ')
            print(ret_simularity)
            print(' ')  
            print('*************************************************')  
        else :
            
            print('*************************************************')
            print('       유사도 비교 방법   :  Matrix Factrazation (ASL)비교 ')
            print('*************************************************')
            
            print('*************************************************')
            print(' Matrix Factrazation 전 matrix ')
            print('*************************************************')
            print(' ')
            print(df_matrix)
            print(' ')  
            print('*************************************************')
            als = ALS(df_matrix)
            ret_simularity = als.get_complete_matrix()
            print(' ')
            print(' ')
            print('*************************************************')
            print(' Matrix Factrazation 적용 matrix ')
            print('*************************************************')
            print(' ')
            print(ret_simularity)
            print(' ')  
            print('*************************************************')  
            
        sel_query = np.array([np.array(self.select_query).T[0]])
# =============================================================================
#         print('selq : ', sel_query)
# =============================================================================
        cosine_sim = cosine_similarity(sel_query, df.values).flatten()
        sim_rank_idx = cosine_sim.argsort()[::-1]
        print(' ')
        print(' ')
        print('*************************************************')
        print(' 유사도 계산 결과 ')
        print('*************************************************')
        print(' ')
        print(sim_rank_idx)
        print(' ')  
        print('*************************************************')  
            
       
        output = []
                
        predict2nd = []
        
        rnkcnt = 30

        dfcomment = pd.DataFrame(self.openFile(self.path_댓글데이터))
        dfcomment['total'] = dfcomment['content']
   
        testCommentKey = ['월계풍림아이원','월계그랑빌','공릉풍림아이원','월계삼창','하계극동건영벽산','월계풍림아이원','상계벽산','월계극동','월계유원','월계삼호4차','중계건영2차','월계서광','공릉1동삼익','공릉대주파크빌','중계건영2차','중계무지개아파트','수락산벨리체아파트','상계미도','월계한일1차']
        testCommentData = {}
    
        item1 = ['석계역과 광운대역 두개의 역을 사용할 수 있는점, 그리고 석계역과 3분거리 라는 점이 강점이죠.',
                   '그랑빌에 비해 구조도 잘 빠졌다는점 확실하구요.',
                   '동부간선로도 가까워서 교통 부분 만족하며 살고 있어요',
                   '풍림 살아보니 참 좋네요 큰차이가 아니라 생각 했었는데 구조가 좋으니 동일평형보다 훨씬크게 느껴지고 초역세권이니 지하철입구까지 너무 가깝네요.',
                   '식당은 많은데 큰 건물이 없어서 대형문화시설이 못들어오는게 별로 인듯',
                   '다소 외진곳에 있어서 택시기사분들이 잘 모를경우 몰라요.',
                   '484세대로 대단지 대비 작지만 요즘 트렌드상 리모델링엔 더 적합하다는 확신이 있었네요']
        
        item2 = ['관리비 실화인가요. 정말 저렴하네요.',
                 '석계역 역세권이고 태릉입구 역도 가까워요. 자차나 대중교통 모두 편리하고 좋습니다.',
                 '대단지라 편의 시설이 다 있어서 좋아요. 인근에 하나로 마트 이마트 있고 중계동 학원가도 멀지 않아오',
                 '차량의 통행이 많아서 어린아이들을 편하게 데리고 다니기는 어렵습니다.',
                 '한천초 초품아 동부간선, 1.6호선 더블역세 교통은 진짜좋음 gtx 광운대 수혜랑 더블역세권 시멘트공장 이전 호재로 살아보고 매수함.',
                 '용적률 높아도 리모델링 불가능 리모델링 이야기가 안나와 아쉬워요',
                 '도로쪽 인근 아파트는 시끄러움. 층간 소음 있어음']
        
        item3 = ['신혼 때 1114동에 살았구요 둘이 알콩달콩 살기엔 최고 였어요',
                   '다만 동남향이고 맞은편 아파트에 해가 가려서 집이 약간 어두운 감은 있어요.',
                   '맞은편 아파트가 살짝 틀어져있어 프라이버시 보호에는 좋아요.',
                   '엘레베이터는 별로 인듯 엘레베이터가 오래되어서 고장이 잦아요. 바꾼다는데 빨리 바뀌면 좋겠습니다. ']
        
        item4 = ['근처에 시장이 있고 대학도 있어서 상권이 좋아요',
                '실거주생각하는데 향후 재건축은 어떻게 될 지 궁금하네요. 재건축 안되면 별론데.',
                '놀이터가 흙이라서 애들키우기는 안좋다',
                '다만 오래된 아파트라 주차 공간이 협소한것은 불편해요',
                '조용학 봄여름가을겨울 너무 예쁜 아파트에요']
        
        testCommentData['월계풍림아이원'] = item1
        testCommentData['월계그랑빌'] = item2
        testCommentData['공릉풍림아이원'] = item3
        testCommentData['월계삼창'] = item4
        
        
        item5 = ['학군좋고 쇼핑센터 많고 1.7호선 가깝고 생활인프라는 최고!!!!',
                   '실거주하기 정말 좋아요. 지하철 1,7호선 가깝고 가까운 영화관, 대형마트, 미술관 다 있고 중랑천과 공원 모두 가까워서 운동하기도 좋아요. 주민들도 조용하고 매너좋고 만족도 높습니다.',
                   '조용하고 아이키우기 환경좋고 살기 좋습니다',
                   '경비실이 출입구와 멀리 떨어져 있어 쓸데없는 접촉이 없고 사생활보호 및 내집 드나들기 편합니다',
                   '중랑천, 편의 시설 가깝고 조용해서 살기 좋아요.']
        item6 = ['석계역과 광운대역 두개의 역을 사용할 수 있는점, 그리고 석계역과 3분거리 라는 점이 강점이죠.',
                   '그랑빌에 비해 구조도 잘 빠졌다는점 확실하구요.',
                   '동부간선로도 가까워서 교통 부분 만족하며 살고 있어요',
                   '풍림 살아보니 참 좋네요 큰차이가 아니라 생각 했었는데 구조가 좋으니 동일평형보다 훨씬크게 느껴지고 초역세권이니 지하철입구까지 너무 가깝네요.',
                   '식당은 많은데 큰 건물이 없어서 대형문화시설이 못들어오는게 별로 인듯',
                   '다소 외진곳에 있어서 택시기사분들이 잘 모를경우 몰라요.',
                   '484세대로 대단지 대비 작지만 요즘 트렌드상 리모델링엔 더 적합하다는 확신이 있었네요']
        item7 = ['지하철 3분이내로 가깝고 초중고 도가까워 살기편합니다 앞으로 동북선 개통되면 집값도오르겠죠~~~^^',
                   '밤에 들어오면 이중주차가 힘들어요.',
                   '지하철 가깝다는것이 정말 최고의 최고의 장점',
                   '첨에 왔을때 바퀴벌레 때문에 소스라친적이 여러번…살면서 바퀴 이렇게 많이 본건 첨이에요. 그리고 관리비가 너무 비쌉니다…ㅠㅠ',
                   '상계역과 인접해서 교통편이 아주 좋고 편의시설도 부족함이 없어요~']
        item8 = ['바로앞에 우이천도 있고 살기 좋은곳 같아요',
                   '층간소음 꽤 있고 주차하기 힘들어요ㅠㅠ',
                   '앞뒤로 산있고 개천있는건 좋아요',
                   '버스정거장 바로앞이라 좋고 101동은 앞쪽 우이천과 뒷쪽 초안산과 맞바람으로 여름에 왠만한 더위빼고 에어콘 필요없음']
        item9 = ['우이천을 끼고 있는 환경이 수채물감으로 그린 풍경화에요. 너무 아름답고, 바로뒤엔 산이 있어서 공기도 좋구요.',
                   '지하주차장도 있고, 입주민들이 다들 온순하신편인듯 해요.',
                   '초안산, 우이천과 접해 있어서 공기가 좋고 산책로가 잘 조정되어 있음',
                   '중간층에 살고 있는데 뷰는 정말 좋아요. 앞에 가리는것이 없는데다 남향이라 햇볕도 잘 들어와요.',
                   '다만 주위 편의시설이 너무 부족해요.']

        testCommentData['하계극동건영벽산'] = item5
        testCommentData['월계풍림아이원'] = item6
        testCommentData['상계벽산'] = item7
        testCommentData['월계극동'] = item8
        testCommentData['월계유원'] = item9

        testCommentData['월계삼호4차'] = ['신혼부부 전세로 좋은곳이라고 봐요 1호선 7호선 도보권이고 이마트도 붙어있어서 좋아요~^^',
               '바로 옆 삼호3차 상가에 하나로마트도 있어서 2,4째 일요일에도 마트 이용 가능',
               '경비아저씨 진짜 친절하시고 동네주민분들도 마주치면 다 인사해주심',
               '주차공간 절대부족 오래되어 모든시설 낙후',
               '광운대역 육교만 넘어가면 코 앞! 종로 쪽 접근성 좋음']
        testCommentData['중계건영2차'] =  ['지하철 1분거리, 롯데슈퍼 1분거리, 아울렛 대형마트 10분 내외, 공원 및 당현천 5분 내외, 정남향,베란다 많아서 좋아요, 스타벅스 맥날 5분 거리',
               '도로측 동은 시끄러워서 문을 못 열어요, 바퀴벌레 많음, 베란다 냄새, 주차 정말 최악, 쓰레기장 별로, 학군 그다지, 먹거리 별로 없음',
               '백날천날 무슨짓을 한다해도 주변 영세민 아파트 싹 다 밀어버리지 않는한(탈북자 포함) 집값은 절대 안오름',
               '역세권,공원,학교,유통점 모두 편리합니다',
               '지하철역이 가까워서 편리하고 주변에 대학병원 마트 영화관등 편의시설이 좋아요.',
               '경비아저씨들 정말 좋으시구요.']
        testCommentData['월계서광'] = ['역 가깝고 살기 좋아요.',
               '전철과교통이마트가 가까이있어서조용하고살기가아주넘좋아요~^^^',
               '조용하고 지하철이 가깝습니다. 이마트도 가깝고 유흥시설이 적어 아이들을 키우기 좋은 동네입니다.',
               '입주민들이 친절하고 편의시설이용하기조아요',
               '하루종일 지하철소리가 시끄러워요 예민 하신분은 못살아요.오래된아파트라 리모델링필수구요.']
        testCommentData['공릉1동삼익'] = ['일단 태릉입구역 6/7호선 역세권이 메리트 있고 바로 앞, 마을 버스 이용하면 석계역도 코앞인 교통과 위치적인 면에선 최고쬬',
               '주변에 시야를 가릴만한 건물이 없어 뷰도 좋습니다',
               '뒤에 동부간선도로가 있어서 그런지 먼지가 많고 앞도로도 버스가 다녀서 여름에 문열어놓고 자기엔 시끄러워요.',
               '주차장 입구가 엄청 좁고 아이들이 놀기엔 놀이터가 조금 음침해요.',
               '한동이라는 아쉬움이좀있어요']
        testCommentData['공릉대주파크빌'] = ['앞이 뻥뚫여있어 중랑천 조망좋고 바로뒤 재래시장 및 공트럴파크가서 간단히 밥먹고 커피마시기 괜찮음',
               '정문이 큰길가에 있어 주차하고 나가기 편하고 근처아파트들에 비해 관리가 잘되있음',
               '음식물쓰레기 처리장치가 지을때부터 설치되어있어 진짜편함',
               '공릉시장 매우 가깝고,차타고 10분이면 이마트, 트레이더스',
               '대규모 단지가 아닌 나홀로 아파트지만 관리 잘해주셔서 깨끗하고 가성비 끝판왕 살기 좋은 아파트입니다 굳bb']
        
        testCommentData['중계건영2차'] = ['이동네서 초중고 다 나왔지만, 생각해보면 질떨어지는 애들은 죄다 목련에 살았음.',
           '도로측 동은 시끄러워서 문을 못 열어요, 바퀴벌레 많음, 베란다 냄새, 주차 정말 최악, 쓰레기장 별로, 학군 그다지, 먹거리 별로 없음',
           '백날천날 무슨짓을 한다해도 주변 영세민 아파트 싹 다 밀어버리지 않는한(탈북자 포함) 집값은 절대 안오름',
           '역세권,공원,학교,유통점 모두 편리합니다',
           '지하철역이 가까워서 편리하고 주변에 대학병원 마트 영화관등 편의시설이 좋아요.',
           '역세권,공원,학교,유통점 모두 편리합니다']
        testCommentData['중계무지개아파트'] = ['양방향으로 마트가 있어서 너무 좋아용',
           '여기 부동산들은 실거래 한달 꼭 채우고 올리기로 했는지..옆 단지보다 무지 느려요...',
           '중계역에서 가깝고, 주변 인프라가 잘 되어있어요.',
           '노원역도 가깝고 중계역에 붙어있고 이래저래 편리했어요. 그리고 층간소음도 못 겪었구요.',
           '평지에 있는 아파트여서 좋아요.',
           '주차가 불편하고, 지하주차장이 없는것 차단기가 없는것은 단점이에요']
        testCommentData['수락산벨리체아파트'] =['주위환경이 조용하고 깨끗이 관리되고 있어요',
           '단지가 조용하고 산과 인접해 있어 공기 좋음',
           '산전망이 좋고 공기는 좋으나 열손실이 많아 굉장히 춥고 관리비가 마니 나와요',
           '주말에는 등산객이 많지만 시끄러울 정도는 아니에요.',
           '다만 시내로 나가려면 1시간은 잡아야합니다.']
        testCommentData['상계미도'] =['평지라서 힘들지 않게 도보 이용이 가능하다',
           '중계역까지 접근성이 좋으며 노원역, 창동역 등 다양한 역으로의 접근이 가능하다.',
           '당현천과 중랑천으로의 접근성이 좋아서 운동하기에도 좋다.',
           '겨울에 정말 너무 추웠어요, 이 부분은 수리를 잘하고 들어오면 된다고들 하시던데 저희는 들어온지 오래된 집이다 보니 겨울에 너무 추웠어요.',
           '서울 끝이다 보니 어딜 출퇴근하나 너무 멀었어요.']
        testCommentData['월계한일1차'] =['아직 가격이 그리 높지않다.',
           '세대수가 적어서 관리비가 싸지 않다는점?',
           '초역세권이라는 말이 딱 어울리는 곳입니다.',
           '대체적으로 출퇴근이 용이해 좋습니다!']

# =============================================================================
#         testCommentData[''] =
#         testCommentData[''] =
#         testCommentData[''] =
#         testCommentData[''] =
#         testCommentData[''] =
# =============================================================================
        
        
        
        
        list_all_maemae_jisu = []

        
        # p1 rank1 계산 
        for idx in sim_rank_idx[:rnkcnt]:
           
            jsonApt = self.json_apt_datas[idx]
                       
# =============================================================================
#             3차 추천용 데이터 뽑기 
# =============================================================================
            aptName = jsonApt['kaptName']
            
            kAptNameSplit = aptName.split()
            
# =============================================================================
#             print("========================================== : ", jsonApt['kaptName'])
# =============================================================================
          
            retDatas = []
            
            isEquals = False
            
            score_p3 = {}
            score_p3[0] = 0
            score_p3[1] = 0
            score_p3[2] = 0
            
            if len(jsonApt['doroJuso'].replace(' ','')) <= 0:
                continue
                    
            # 실거래가 금액 뽑아보자 
            juso_split = jsonApt['doroJuso'].split()
            
        
            print(jsonApt)
# =============================================================================
#             print('주소 :', jsonApt['doroJuso'])
#             print('주소 split 결과 :', juso_split)
# =============================================================================

            doroname = ''            
            if len(juso_split[3]) > 0:                
                doroname = juso_split[2] + ' ' +juso_split[3]
                doroname = doroname.strip()
                
            else :
                doroname = juso_split[2]
            

            dfAptInfo = self.df실거래가[
                    (self.df실거래가['도로명'] == doroname) &
                    (self.df실거래가['거래금액(만원)'].astype(int) <= int(self.myprice))
                    ]
            
            if len(dfAptInfo) <= 0:
                # 아파트 정보가 없다면 pass
# =============================================================================
#                 print('아파트 정보 없음 ')
# =============================================================================
                continue
            
            queryDataEqualCounts.append(jsonApt['queryDataEqualCount'])
            queryDataNotSelItemCnt.append(jsonApt['queryDataNotSelItemCnt'])
            queryDataTotalkm.append(jsonApt['queryDataTotalkm'])

# =============================================================================
#             print("-----------------------------------------------")            
#             print(dfAptInfo['거래금액(만원)'].astype(int).idxmax())
#             print(dfAptInfo.iloc(3289))
#             print("-----------------------------------------------")            
#             
# =============================================================================

            maxPrice = str(dfAptInfo['거래금액(만원)'].astype(int).max())
            minPrice = str(dfAptInfo['거래금액(만원)'].astype(int).min())
            
# =============================================================================
#             maxDf = dfAptInfo[dfAptInfo['거래금액(만원)'].astype(int) == maxPrice]
#             minDf = dfAptInfo[dfAptInfo['거래금액(만원)'].astype(int) == minPrice]
#             
# =============================================================================
# =============================================================================
#             jisuItem['max_room_size'] = str(maxDf['전용면적(㎡)'])[:2]
#             jisuItem['min_room_size'] = str(minDf['전용면적(㎡)'])[:2]
#             
# =============================================================================
            jsonApt['max_price'] = maxPrice
            jsonApt['min_price'] = minPrice

            maemae_jisu = []

# =============================================================================
# 노원구 아닐경우 실거래가만 표기             
# =============================================================================
            if '노원구' not in self.guName:

                recentRealPrice = ''
                recentRoomSize = ''
                maxRoomSize = ''
                minRoomSize = ''
                for i, d in dfAptInfo.iterrows():                  
                     jisuItem = {}
                     jisuItem['date'] = str(int(d['계약년월']))
                     jisuItem['realprice'] = str(int(d['거래금액(만원)']))
                     jisuItem['predict_jisu'] = []
                     
                     room_size = str(d['전용면적(㎡)'])


                     jisuItem['room_size'] = room_size
                     
                     if int(d['거래금액(만원)']) >= int(maxPrice):
                         maxRoomSize = room_size
                     if int(d['거래금액(만원)']) <= int(minPrice):
                         minRoomSize = room_size
                         
                     maemae_jisu.append(jisuItem);

                
                maemae_jisu = sorted(maemae_jisu, key=(lambda maemae_jisu:maemae_jisu['date']), reverse = False)  
                maemae_jisu_lastItem = maemae_jisu[len(maemae_jisu) - 1]
                jsonApt['maemae_jisu'] = maemae_jisu  
                jsonApt['recent_price'] = maemae_jisu_lastItem['realprice']                
                jsonApt['recent_room_size'] = maemae_jisu_lastItem['room_size']                   
                jsonApt['max_price_room_size'] = maxRoomSize                  
                jsonApt['min_price_room_size'] = minRoomSize              
                
                jsonApt['comment'] = []
                
                output.append(jsonApt)
                
            else:
                
                print('*************************************************')
                print('*                     2차 추천                   *')
                print('*************************************************')
                                
                predict = 0        
                
                testkey = ['계약년월', '거래금액(만원)', '매매가격지수', '통화금융지표', '기준금리']
# =============================================================================
#                 dfAptInfo2 = dfAptInfo[dfAptInfo['계약년월'].str.contains("2020") == False]
# =============================================================================
                dfAptInfo2 = dfAptInfo[testkey]
                
# =============================================================================
#                 predict = self.clf_from_joblib.predict(dfAptInfo2)
# =============================================================================
                maemae_jisu_lastItem = {}
                maemae_jisu_2020 = []
                index = 0
                recentRoomSize = ''
                maxRoomSize = ''
                minRoomSize = ''
                
                test = []
                for i, d in dfAptInfo.iterrows():                  
                    strYYYY = str(d['계약년월'])[0:4]
                    yyyy = int(strYYYY)
                    
                    room_size = str(d['전용면적(㎡)'])
# =============================================================================
#                     room_size_comma_idx = room_size.index('.')
# =============================================================================
                    if int(d['거래금액(만원)']) >= int(maxPrice):
                         maxRoomSize = room_size
                         
                    if int(d['거래금액(만원)']) <= int(minPrice):
                         minRoomSize = room_size
                         
                    if yyyy < 2020:                            
                         jisuItem = {}
                         jisuItem['date'] = str(int(d['계약년월']))
                         jisuItem['jisu'] = str(d['매매가격지수'])
                         jisuItem['realprice'] = str(int(d['거래금액(만원)']))
                         jisuItem['room_size'] = room_size

    # =============================================================================
    #                      jisuItem['predict_jisu'] = str(predict[index])
    # =============================================================================
                         jisuItem['predict_jisu'] = str(self.clf_from_joblib.predict([[d['계약년월'],d['거래금액(만원)'],d['매매가격지수'],d['통화금융지표'],d['기준금리']]])[0])
                         maemae_jisu.append(jisuItem);
                         maemae_jisu_lastItem = d
                    else :
                         jisuItem = {}
                         jisuItem['date'] = str(int(d['계약년월']))
                         jisuItem['realprice'] = str(int(d['거래금액(만원)']))     
                         jisuItem['room_size'] = room_size                                 
                         maemae_jisu_2020.append(jisuItem)

                        
# =============================================================================
#                     print('[',room_size[:room_size_comma_idx],']  ', jisuItem['realprice'])
# =============================================================================
# =============================================================================
#                 recentRoomSize = str(dfAptInfo.iloc[i]['전용면적(㎡)'])
# =============================================================================
                print("1234567 : ", test)
                
                maemae_jisu = sorted(maemae_jisu, key=(lambda maemae_jisu:maemae_jisu['date']), reverse = False)   
                maemae_jisu_2020 = sorted(maemae_jisu_2020, key=(lambda maemae_jisu_2020:maemae_jisu_2020['date']), reverse = False)   
                                
# =============================================================================
#                 print('* test set :', set(test))
# =============================================================================
    # =============================================================================
    #                     if not maemae_jisu_lastItem:
    # =============================================================================
                maemaelen = len(maemae_jisu)
                
                if maemaelen > 0 :
                    maemae_jisu_lastItem = maemae_jisu[len(maemae_jisu) - 1]
        
                    # 나중에 minmax사용하기 위해 사용 
                    list_all_maemae_jisu.append(float(maemae_jisu_lastItem['predict_jisu']))
                    
                    jsonApt['max_price_room_size'] = maxRoomSize                  
                    jsonApt['min_price_room_size'] = minRoomSize   
                    
                    if len(maemae_jisu_2020) > 0:
                        lastitem_2020 = maemae_jisu_2020[len(maemae_jisu_2020) - 1]
                        jsonApt['recent_price'] = int(lastitem_2020['realprice'])
                        jsonApt['recent_room_size'] = lastitem_2020['room_size']
                        
                    else:
                        jsonApt['recent_price'] = int(maemae_jisu_lastItem['realprice'])    
                        jsonApt['recent_room_size'] = maemae_jisu_lastItem['room_size']
                    
                    # 데이터에서 가장 최근의 날짜를 가져온다.
                    yyyyMM = maemae_jisu_lastItem['date']
                
                    # 가장 마지막 데이터를 가져온다. 
                    last2NdItem = dfAptInfo2[dfAptInfo2['계약년월'].astype(str).str.contains(yyyyMM)]
                    
                    print("가장 최근의 실거래가 데이터 추출 ", last2NdItem)
        
                    from dateutil.relativedelta import relativedelta
                                       
                    def createPredict2NdDataAfterMonth(lastItem, afterMonth):
                         maemaejisu ={}
    
                         nextMonth = (datetime.strptime(str(int(lastItem['계약년월'].iloc[0])), '%Y%m')+ relativedelta(months=afterMonth)).strftime("%Y%m")
                         maemaejisu['계약년월'] = nextMonth
                         
                         ret = {}
                         ret['date'] = nextMonth
    
                         ret['predict_jisu'] = str(self.clf_from_joblib.predict(lastItem)[0])
    
                         return ret
                    
                    # 1개월 뒤만 표기 
                    predict2ndAfter1MonthData = createPredict2NdDataAfterMonth(last2NdItem, 1)
                    
                    print('*************************************************')
                    print('*           2차 매개가격지수 1달뒤 예측               ', predict2ndAfter1MonthData['predict_jisu'])
                    print('*************************************************')
                    maemae_jisu.append(predict2ndAfter1MonthData)

                jsonApt['maemae_jisu'] = maemae_jisu
                jsonApt['maemae_jisu_2020'] = maemae_jisu_2020
                
    # =============================================================================
    #                     for i in range(1,13):
    #                         maemae_jisu.append(createPredict2NdDataAfterMonth(last2NdItem.iloc[0], i))
    # # =============================================================================
    # =============================================================================
    #                     maemae_jisu.append(createPredict2NdDataAfterMonth(last2NdItem.iloc[0], 2))
    #                     maemae_jisu.append(createPredict2NdDataAfterMonth(last2NdItem.iloc[0], 3))
    #                     
    # =============================================================================
    
    
# =============================================================================
#                 if int(self.optSecond) == 1 and '노원구' in self.guName:
# =============================================================================
                print('*************************************************')
                print('*                     3차 추천                   *')
                print('*************************************************')

                if aptName in testCommentKey:
                
                    for d in testCommentData[aptName]:                        
                        emotion = {}
                        emotion['content'] = d
    
                        label = multi_nb.predict([d])[0]
                        emotion['predict'] = str(label)
                                            
                        score_p3[label] = score_p3[label] + 1
                                                
                        retDatas.append(emotion)
                        isEquals = False
                else :
                    
                    if len(kAptNameSplit) > 0:    
        
                        for split in kAptNameSplit:
                            dftmp = dfcomment[dfcomment['total'].str.contains(split)]
                                                
                            if len(dftmp) != 0:
                                isEquals = True
                            else:
                                isEquals = False
                                break
                
                        if isEquals:
                            
                            for i, d in dftmp.iterrows():                        
                                emotion = {}
                                emotion['content'] = d['content']
                               # print(d['text_comment'])
                                label = multi_nb.predict([d['content']])[0]
                                emotion['predict'] = str(label)
                                
                                score_p3[label] = score_p3[label] + 1
                                                        
                                retDatas.append(emotion)
                                isEquals = False
        
                    else :
                        for i,d in dfcomments.iterrows():
                            
                            emotion = {}
                            emotion['content'] = d['content']
                            
                            label = multi_nb.predict([d['content']])[0]
                            emotion['predict'] = str(label)
                            score_p3[label] = score_p3[label] + 1
                            retDatas.append(emotion)
    
                try:
                    pos_cnt = score_p3[2] 
                    neg_cnt = score_p3[0]
                    
                    if pos_cnt > neg_cnt:
                        # 긍정
                        max_score_p3 = 10
                        
                    elif pos_cnt < neg_cnt:
                        # 부정 
                        max_score_p3 = 5
                        
                    elif pos_cnt == neg_cnt:
                        max_score_p3 = 0
                        
                    else:
                        #없으경우 같을경우
                        max_score_p3 = -1

                except:
                    max_score_p3 = -1
                    
                jsonApt['comment'] = retDatas
                jsonApt['score_p3'] = str(max_score_p3)
# =============================================================================
#                 print(score_p3)
#                 print(max_score_p3)
# =============================================================================
                                
                # score 기본값 할당 
                jsonApt['score'] = 0

                  
                output.append(jsonApt)
            
# =============================================================================
#         print('* 2nd predict? : ',int(self.optSecond))
# =============================================================================


        # 1차는 300점 만점
        # 내가 선택한 아이템과 일치한가? 100점
        # 내가 선택한 아이템의 거리가 가장 작은가? 100점
        # 내가 선택한 아이템중 가장 많은 주변시설을 가지고 있는가? 100점
# =============================================================================
#         print('111 : ', queryDataEqualCounts)
#         print('222 : ', queryDataNotSelItemCnt)
#         print('333 : ', queryDataTotalkm)
# =============================================================================
        maxQueryDataEqualCounts = 0
        try:
            maxQueryDataEqualCounts = max(queryDataEqualCounts)
        except:
            pass
        
        minQueryDataTotalkm = 0
        try:
            minQueryDataTotalkm = min(queryDataTotalkm)
        except:
            pass
        
        maxQueryDataNotSelItemCnt = 0
        try:
            maxQueryDataNotSelItemCnt = max(queryDataNotSelItemCnt)
        except:
            pass
       
        p1_max_score = 300
        
        if maxQueryDataEqualCounts != 0 and minQueryDataTotalkm != 0:
            
            for out in output:
                
                if maxQueryDataNotSelItemCnt == 0:
                    
                    p1_max_score = 200
    

                p1_con1 = (out['queryDataEqualCount'] / queryTrueValueCnt) * 100
                print(out['queryDataEqualCount'], ' / ', queryTrueValueCnt, )
                p1_con2 = (out['queryDataTotalkm'] / minQueryDataTotalkm) * 100
                
                if p1_con2 > 100:
                   p1_con2 = 200 - p1_con2
                           
                p1_con3 = 0            
                if maxQueryDataNotSelItemCnt != 0:
                    p1_con3 = (out['queryDataNotSelItemCnt'] / maxQueryDataNotSelItemCnt) * 100
            
                if p1_max_score == 300:
                    sort_score = ((p1_con1 + p1_con2 + p1_con3) / 300) * 100                
                else :
                    sort_score = ((p1_con1 + p1_con2) / 200) * 100                
                
                
                out['score_sim'] = str(round(p1_con1,2))
                out['score_p1'] = str(int(p1_con1))
                out['score_sort'] = float(sort_score)
                
        else :
            
            max_p1_score = 100
            for out in output:
                
                out['score_sim'] = str('-')
                out['score_p1'] = str(max_p1_score)
                out['score_sort'] = float(max_p1_score)
                max_p1_score -= 5
            

        finalScore = []
                
        if int(self.optSecond) == 1 and '노원구' in self.guName:
            maxjisu = 105.74
            minjisu = 85.28
            
            for i, d in enumerate(output):
                try:
                    mmjisu = d['maemae_jisu']
                    lastmmjisu = mmjisu[len(mmjisu) - 1]
                    #'scorep2 매매가격지수 : ', 
    # =============================================================================
    #                 print(float(lastmmjisu['predict_jisu']))
    # =============================================================================
                    jisuscale = (float(lastmmjisu['predict_jisu']) - minjisu) / (maxjisu - minjisu)
                    
                    scorep2 = jisuscale * 40
                    
                    d['score_p2'] = str(int(scorep2))
                    
                    scorep3 = 0
                    if d['score_p3'] != '0':
                        scorep3 = d['score_p3']
                        d['score_p3_len'] = d['score_p3']
                        
                    else :
                        d['score_p3_len'] = 0
    
                        
                    d['score'] = float(d['score_p1']) + float(scorep2) + float(scorep3)
                    finalScore.append(d)

                except:
                    pass
            
            result = sorted(finalScore, key=(lambda finalScore:(finalScore['score'])), reverse = True)                
           


# =============================================================================
#             
# =============================================================================
            return result        
        else:
            
            for d in output:
                d['score_p2'] = 0
                d['score_p3'] = 0
                d['score'] = d['score_p1']
                d['score_p3_len'] = 0
                
                finalScore.append(d)
            
            result = sorted(finalScore, key=(lambda finalScore:(finalScore['score_p1'], finalScore['score_sort'])), reverse = True)                
            return result
            
# =============================================================================
#         print(result)
# =============================================================================


    def getDistanceKmPointToPoint(self, toLat, toLon, fromLat, fromLon):
        toLoc = (float(toLat), float(toLon))
        fromLoc = (float(fromLat), float(fromLon))
        km = haversine(toLoc, fromLoc)
        return km

    def getInRangeSelectData(self, aptData, selectItems, itemLimitKm):    
        # 위도 기준으로는 + 0.01이 약 +1km에 해당하고, 경도 기준으로는 +0.015가 약 +1km 정도에 해당합니다.
        # 범위 계산
        std_lat_wgs_limit_min = float(aptData['LAT_WGS']) - ( 0.01 * itemLimitKm )
        std_lat_wgs_limit_max = float(aptData['LAT_WGS']) + ( 0.01 * itemLimitKm )

        std_lon_wgs_limit_min = float(aptData['LON_WGS']) - ( 0.015 * itemLimitKm )
        std_lon_wgs_limit_max = float(aptData['LON_WGS']) + ( 0.015 * itemLimitKm )

        # 해당 범위 안에 들어오는 아이템들만 추린다. 
        inRangeItems = [] 
        
        selQueryInRangeItemTotalkm = 0
        
        for item in selectItems:
            try:

                lat = float(item['LAT_WGS'])
                lon = float(item['LON_WGS'])

                if (std_lat_wgs_limit_min <= lat and std_lat_wgs_limit_max >= lat):
                    if(std_lon_wgs_limit_min <= lon and std_lon_wgs_limit_max >= lon):
                        # 직선 거리 구한다.                     
                        km = self.getDistanceKmPointToPoint(float(aptData['LAT_WGS']), float(aptData['LON_WGS']), lat, lon)
                        item['km'] = km
    
                        if km <= itemLimitKm:   
                            selQueryInRangeItemTotalkm += km
                            inRangeItems.append(item)

            except:
# =============================================================================
#                 print(item)
# =============================================================================
                pass
                 


        return inRangeItems, selQueryInRangeItemTotalkm
    
        
    ######################################
    #
    # 데이터 로드
    #
    ######################################
    def openFile(self, path):    
        with open(path, 'r', encoding='UTF8') as f:
            datas = json.load(f)
        return datas




import numpy as np

class ALS():
    
    def __init__(self, matrixR):
        self.r_lambda = 0.1
        self.nf = 200
        self.alpha = 0.1

        self.nu = matrixR.shape[0]
        self.ni = matrixR.shape[1]
        self.R = matrixR
        # initialize X and Y with very small values
        self.X = np.random.rand(self.nu, self.nf) * 0.01
        self.Y = np.random.rand(self.ni, self.nf) * 0.01
        # Initialize Binary Rating Matrix P
        self.P = np.copy(matrixR)
        self.P[self.P > 0] = 1
        #Initialize Confidence Matrix C¶
        self.C = 1 + self.alpha * matrixR
    
    def loss_function(self, C, P, xTy, X, Y, r_lambda):
        predict_error = np.square(P - xTy)
        confidence_error = np.sum(C * predict_error)
        regularization = r_lambda * (np.sum(np.square(X)) + np.sum(np.square(Y)))
        total_loss = confidence_error + regularization
        return np.sum(predict_error), confidence_error, regularization, total_loss
    
    def optimize_user(self, X, Y, C, P, nu, nf, r_lambda):
        yT = np.transpose(Y)
        for u in range(nu):
            Cu = np.diag(C[u])
            yT_Cu_y = np.matmul(np.matmul(yT, Cu), Y)
            lI = np.dot(r_lambda, np.identity(nf))
            yT_Cu_pu = np.matmul(np.matmul(yT, Cu), P[u])
            X[u] = np.linalg.solve(yT_Cu_y + lI, yT_Cu_pu)

    def optimize_item(self, X, Y, C, P, ni, nf, r_lambda):
        xT = np.transpose(X)
        for i in range(ni):
            Ci = np.diag(C[:, i])
            xT_Ci_x = np.matmul(np.matmul(xT, Ci), X)
            lI = np.dot(r_lambda, np.identity(nf))
            xT_Ci_pi = np.matmul(np.matmul(xT, Ci), P[:, i])
            Y[i] = np.linalg.solve(xT_Ci_x + lI, xT_Ci_pi)
    
    def get_complete_matrix(self):
        predict_errors = []
        confidence_errors = []
        regularization_list = []
        total_losses = []
        
        for i in range(5):
    
            if i!=0:   
                self.optimize_user(self.X, self.Y, self.C, self.P, self.nu, self.nf, self.r_lambda)
                self.optimize_item(self.X, self.Y, self.C, self.P, self.ni, self.nf, self.r_lambda)
            predict = np.matmul(self.X, np.transpose(self.Y))
            predict_error, confidence_error, regularization, total_loss = self.loss_function(self.C, self.P, predict, self.X, self.Y, self.r_lambda)
            
            predict_errors.append(predict_error)
            confidence_errors.append(confidence_error)
            regularization_list.append(regularization)
            total_losses.append(total_loss)
            
            print('----------------step %d----------------' % i)
            print("predict error: %f" , predict_error)
            print("confidence error: %f" , confidence_error)
            print("regularization: %f" , regularization)
            print("total loss: %f" , total_loss)
            
        
# =============================================================================
#         predicts ={}
#         predicts['predict_error'] = predict_errors
#         predicts['confidence_error'] = confidence_errors
#         predicts['regularization'] = regularization_list
#         predicts['total_loss'] = total_losses
#         
#         import json
#         with open('first_predict_graph.json', 'w', encoding='utf-8') as make_file:
#             json.dump(predicts, make_file, indent="\t")
# =============================================================================
        
        predict = np.matmul(self.X, np.transpose(self.Y))
        print('final predict')
        print([predict])
        return predict


class MatrixFactorization():
    def __init__(self, R, k, learning_rate, reg_param, epochs, verbose=False):
        """
        :param R: rating matrix
        :param k: latent parameter
        :param learning_rate: alpha on weight update
        :param reg_param: beta on weight update
        :param epochs: training epochs
        :param verbose: print status
        """

        self._R = R
        self._num_users, self._num_items = R.shape
        self._k = k
        self._learning_rate = learning_rate
        self._reg_param = reg_param
        self._epochs = epochs
        self._verbose = verbose


    def fit(self):
        """
        training Matrix Factorization : Update matrix latent weight and bias

        참고: self._b에 대한 설명
        - global bias: input R에서 평가가 매겨진 rating의 평균값을 global bias로 사용
        - 정규화 기능. 최종 rating에 음수가 들어가는 것 대신 latent feature에 음수가 포함되도록 해줌.

        :return: training_process
        """

        # init latent features
        self._P = np.random.normal(size=(self._num_users, self._k))
        self._Q = np.random.normal(size=(self._num_items, self._k))

        # init biases
        self._b_P = np.zeros(self._num_users)
        self._b_Q = np.zeros(self._num_items)
        self._b = np.mean(self._R[np.where(self._R != 0)])

        # train while epochs
        self._training_process = []
        
        savedata=[]
        for epoch in range(self._epochs):

            # rating이 존재하는 index를 기준으로 training
            for i in range(self._num_users):
                for j in range(self._num_items):
                    if self._R[i, j] > 0:
                        self.gradient_descent(i, j, self._R[i, j])
            cost = self.cost()
            self._training_process.append((epoch, cost))
            savedata.append(cost)
            # print status
            if self._verbose == True and ((epoch + 1) % 10 == 0):
                print("Iteration: %d ; cost = %.4f" % (epoch + 1, cost))
                
                
# =============================================================================
#         import json
#         with open('first_gd_predict_graph.json', 'w', encoding='utf-8') as make_file:
#             json.dump(savedata, make_file, indent="\t")
# =============================================================================


    def cost(self):
        """
        compute root mean square error
        :return: rmse cost
        """

        # xi, yi: R[xi, yi]는 nonzero인 value를 의미한다.
        # 참고: http://codepractice.tistory.com/90
        xi, yi = self._R.nonzero()
        predicted = self.get_complete_matrix()
        cost = 0
        for x, y in zip(xi, yi):
            cost += pow(self._R[x, y] - predicted[x, y], 2)
        return np.sqrt(cost) / len(xi)


    def gradient(self, error, i, j):
        """
        gradient of latent feature for GD

        :param error: rating - prediction error
        :param i: user index
        :param j: item index
        :return: gradient of latent feature tuple
        """

        dp = (error * self._Q[j, :]) - (self._reg_param * self._P[i, :])
        dq = (error * self._P[i, :]) - (self._reg_param * self._Q[j, :])
        return dp, dq


    def gradient_descent(self, i, j, rating):
        """
        graident descent function

        :param i: user index of matrix
        :param j: item index of matrix
        :param rating: rating of (i,j)
        """

        # get error
        prediction = self.get_prediction(i, j)
        error = rating - prediction

        # update biases
        self._b_P[i] += self._learning_rate * (error - self._reg_param * self._b_P[i])
        self._b_Q[j] += self._learning_rate * (error - self._reg_param * self._b_Q[j])

        # update latent feature
        dp, dq = self.gradient(error, i, j)
        self._P[i, :] += self._learning_rate * dp
        self._Q[j, :] += self._learning_rate * dq


    def get_prediction(self, i, j):
        """
        get predicted rating: user_i, item_j
        :return: prediction of r_ij
        """
        return self._b + self._b_P[i] + self._b_Q[j] + self._P[i, :].dot(self._Q[j, :].T)


    def get_complete_matrix(self):
        """
        computer complete matrix PXQ + P.bias + Q.bias + global bias

        - PXQ 행렬에 b_P[:, np.newaxis]를 더하는 것은 각 열마다 bias를 더해주는 것
        - b_Q[np.newaxis:, ]를 더하는 것은 각 행마다 bias를 더해주는 것
        - b를 더하는 것은 각 element마다 bias를 더해주는 것

        - newaxis: 차원을 추가해줌. 1차원인 Latent들로 2차원의 R에 행/열 단위 연산을 해주기위해 차원을 추가하는 것.

        :return: complete matrix R^
        """
        return self._b + self._b_P[:, np.newaxis] + self._b_Q[np.newaxis:, ] + self._P.dot(self._Q.T)


    def print_results(self):
        """
        print fit results
        """

        print("User Latent P:")
        print(self._P)
        print("Item Latent Q:")
        print(self._Q.T)
        print("P x Q:")
        print(self._P.dot(self._Q.T))
        print("bias:")
        print(self._b)
        print("User Latent bias:")
        print(self._b_P)
        print("Item Latent bias:")
        print(self._b_Q)
        print("Final R matrix:")
        print(self.get_complete_matrix())
        print("Final RMSE:")
        print(self._training_process[self._epochs-1][1])




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 10:09:34 2020

@author: a60067648
"""
from flask_cors import CORS
from flask import Flask, request, jsonify

app = Flask (__name__)
CORS(app)

recommend = Recommend1()
    
@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/hello')
def hello_world2():
    return 'Hello, World2!'
 
@app.route('/user/<userName>') # URL뒤에 <>을 이용해 가변 경로를 적는다
def hello_user(userName):
    return 'Hello, %s'%(userName)

@app.route('/send_py', methods = ['POST'])
def userLogin():
#
    d = request.form.to_dict()
    print('algo : ', d['algo'])
    print('gu : ', d['gu'])
    print('optSecond : ', d['optSecond'])


    print('선택조건 1 초등학교 선택여부 : ', d['select1'])
    print('선택조건 2 중하교 선택여부 : ', d['select2'])
    print('선택조건 3 고등학교 선택여부 : ', d['select3'])
    print('선택조건 4 학원 선택여부 : ', d['select4'])
    print('선택조건 5 마트 선택여부 : ', d['select5'])
    print('선택조건 6 백화점 선택여부 : ', d['select6'])
    print('선택조건 7 지하철 선택여부 : ', d['select7'])
    print('선택조건 8 병원  선택여부 : ', d['select8'])
    print('선택조건 9 공원  선택여부 : ', d['select9'])
    print('최대 자산정보 : ', d['myprice'])
       
    recommend.setup( d['algo'], d['gu'], d['optSecond'], d['select1'], d['select2'],d['select3'],d['select4'],d['select5'],d['select6'],d['select7'],d['select8'],d['select9'],d['myprice'])
    out = recommend.run(multi_nbc)
    
    data = {}
    data['data'] = out
    ojson = jsonify(data)
    print('output : ', ojson)
    return ojson
#     return jsonify(user)# 받아온 데이터를 다시 전송
# 엑셀파일 불러옴 
import pandas as pd
import sys

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from time import time

from konlpy.tag import Mecab
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


stopword = [
'이','는','가','에','하','은','도','있','을','들','네요',
'으로','의','습니다','한','것','로','를','지','게','에서',
'년', '입니다','는데','어','기','억','었','시','더','합니다','적',
'과','겠','나','해','라','까지','인','그','죠',
'보다','살','니','된','저','해서',
'았','와','차','요','라고','듯',
'ㄷ','길','받','신','곳','다는','라는','데','높','화','던','초','및','서','아요',
'건','동','사',
'그리고','싶','오','여',
'어서','어요','인데','아서','이제','보이','으면','아직','은데']


multi_nbc = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2),stop_words=stopword,
                                               )),
                      ('nbc', MultinomialNB())])



# =============================================================================
# mecab = Mecab()
# 
# def tokenizer_mecab_morphs(doc):
#     return mecab.morphs(doc)
# 
# 
# multi_nbc = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2),stop_words=stopword, tokenizer=tokenizer_mecab_morphs)),
#                       ('nbc', MultinomialNB())])
# =============================================================================



if __name__ == "__main__":
       
    df = pd.DataFrame()
    
    xlspath = 'datas/labelData.xlsx'
    df1_1 = pd.read_excel(xlspath,sheet_name='4-1.내용-중립제거(2019,1~500)')
    
    df2_1 = df1_1[['문장분류','긍정-2, 중립-1, 부정-0']]
    
    df2_1 = df2_1.dropna()
    
    df = df1_1

    train = []
    label = []
    for i, d in df.iterrows():
        # 중립 제거
        # ,(콤마) 제거 
        l = int(d['긍정-2, 중립-1, 부정-0'])
        if l != 1:        
            data = d['문장분류'][d['문장분류'].find(',') + 1:]
            train.append(data)
            label.append(int(d['긍정-2, 중립-1, 부정-0']))
    
    from sklearn.model_selection import train_test_split
    
    n = 5
    kfold = KFold(n_splits=n, shuffle=True, random_state=0)
    scores = cross_val_score(multi_nbc, train, label, cv=kfold)
    print('\n')
    print('\n')
    print('n_splits={}, 3차 cross validation score: {}'.format(n, scores))
    print('\n')
    print('\n')
    multi_nbc.fit(train, label)

    app.run(host='0.0.0.0',port=5001)

# =============================================================================
#     recommend.setup('3', '노원구', '0', '1', '1','0','0','0','0','0','0','0','3000000000')
#     out = recommend.run(multi_nbc)
# =============================================================================
 
# =============================================================================
#     data = {}
#     data['data'] = out
#     ojson = jsonify(data)
#     
# =============================================================================
    
    
    
    

# =============================================================================
# 구구분 = '노원구'
# 도로명 = '섬밭로 232'
# 
# dfgu = df[(df['시군구'].str.contains(구구분)) & (df['도로명'] == 도로명)]
# dfgu.head()
# 
# import json
# def openFile(path):    
#     with open(path, 'r') as f:
#         datas = json.load(f)
#     return datas
# 
# dataApt노원구 = openFile('apt_data_노원구.json')
# len(dataApt노원구)
# 
# from pandas import DataFrame
# 
# listapt = []
# 
# dfApt노원구 = DataFrame(dataApt노원구)
# dfApt노원구['도로명'] = dfApt노원구['doroJuso'].str.replace('서울특별시 노원구 ', '')
# 
# dfTotal = dfgu[dfgu['도로명'].isin(dfApt노원구['도로명']) ]
# 
# len(dfTotal)
# 
# =============================================================================


# =============================================================================
#     X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0.1)
# =============================================================================

# =============================================================================
# 
#     start = time()
#     multi_nbc.fit(X_train, y_train)
#     end = time()
# #     print('Time: {:f}s'.format(end-start))
#     y_pred = multi_nbc.predict(X_test)
#     accScore = accuracy_score(y_test, y_pred)
#     print("3차 테스트 테스트 정확도: {:.3f}".format(accScore))
# =============================================================================

# =============================================================================
#     def getRecommend3rd(self, searchword):
#         
#         print('recommend word : ', searchword)
#         
#         self.driver.get('https://cafe.naver.com/jaegebal')
#         
#         ret = []
#         self.driver.find_element_by_xpath('//*[@id="topLayerQueryInput"]').send_keys(searchword)
#         self.driver.find_element_by_xpath('//*[@id="cafe-search"]/form/button').click()
#         
#         ts.sleep(0.3)
#         
#         totalListCnt = 1
#         listItemCnt = 1
#         pageNoCnt = 1
#         
#         # f = open('result20200523.csv','a', newline='\n')
#         # wr = csv.writer(f)
#         
#         loopFlag = True
#         # try:
#         listData = []
#         while loopFlag:    
#         
#             try :
#                 self.driver.switch_to.frame("cafe_main")
#             except:
#                 pass
#         
#             totalListCnt = len(self.driver.find_elements_by_xpath('//*[@id="main-area"]/div[5]/table/tbody/tr'))
#         
#             # 인덱스 초기화 
#             if listItemCnt > totalListCnt:
#                 totalListCnt = 0
#                 listItemCnt = 1
#         
#                 ######################################
#                 #
#                 # Page 컨트롤 
#                 #
#                 ######################################
#                 try:
#                     totalPageNoCnt = len(self.driver.find_element_by_class_name('prev-next').find_elements_by_tag_name('a'))
#                 except:
#                     totalPageNoCnt = 1
#                     
#                 # 인덱스 초기화 
#                 items = self.driver.find_element_by_class_name('prev-next')
#                 items = items.find_elements_by_tag_name('a')
#                 
#                 try:
#                     clickItem = self.driver.find_element_by_xpath('//*[@id="main-area"]/div[7]/a['+str(pageNoCnt)+']')
#                 except:
#                     loopFlag = False
#                     print('loop 종료 ')
#                     continue
#                     
#                 clickItemText = clickItem.text
#         
#                 clickItem.click()
#                 ts.sleep(0.5)
#                 pageNoCnt += 1
#             #     print('pageCnt : ', pageNoCnt, ' totalPageNoCnt : ' ,totalPageNoCnt)
#         
#                 if '다음' in clickItemText:
#                     # 다음으로 넘어간 페이지 +1, 이전버튼 +1 해서 이전버튼이 있다면 3부터 인덱스 시작함. 
#                     pageNoCnt = 3
#         
#                 else:
#                     # 전체 페이지카운트 보다 현재 리스트 카운트가 더 크다면 루프 종료
#                     if pageNoCnt > totalPageNoCnt:
#                         loopFlag = False
#                         print('loop 종료 ')
#                         break
# 
#             try:
#                 item = self.driver.find_element_by_xpath('//*[@id="main-area"]/div[5]/table/tbody/tr['+str(listItemCnt)+']/td[1]/div[2]/div/a[1]')
#         
#                 # 제목 
#                 title = item.text
#                 print(title)
#                 ret.append(title)
#                 listItemCnt += 1
#             except:
#                 loopFlag = False
#                 print('loop 종료 ')
# 
#         
#         return ret
# =============================================================================