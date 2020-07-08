# -*- coding: utf-8 -*-
# 라이브러리 import
import json
import pandas as pd
import numpy as np
from haversine import haversine, Unit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

import torch
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
import joblib

from datetime import datetime
        
class Recommend1:
    
    path_아파트 = 'datas/apt_list_seoul/apt_data_노원구.json'
    path_댓글데이터 = 'datas/naver_comments/result_contents_네이버카페_노원구.json'
        
    path_초등학교 = 'datas/data_ele'
    path_중학교 = 'datas/data_mid'
    path_고등학교 = 'datas/data_high'

    path_학원 = 'datas/data_edu.txt'
    path_마트 = 'datas/data_mart'
    path_백화점 = 'datas/data_deptment' 
    path_지하철 = 'datas/data_subway'
    
    path_병원 = 'datas/data_hosp'
    path_공원 = 'datas/data_park'
    
    path_실거래가2 = 'datas/dataApt실거래가_노원구.csv'
    path_실거래가 = 'datas/dataApt실거래가_노원구.json'
     
    path_2ndModel = 'datas/20200708_recomment2nd.pkl'
    path_3ndModel = 'C:/Users/pyk12/Downloads/recommendModel3rd_model'
    
    def loadData(self):
        # 노원구 시설 데이터 로드 
        self.json_apt_datas = self.openFile(self.path_아파트)

        
        
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
        active_limitkm = 1
        inactive_limitkm = 1
        
        sel = float(select)
        
        if sel > 0.0:
            return [sel,active_limitkm]
        else:
            return [sel,inactive_limitkm]
    
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
    
    def __init__(self):
        
# =============================================================================
#         self.device = torch.device('cpu') 
#         self.model = torch.load(self.path_3ndModel, map_location=self.device)
#         self.model.eval()
# =============================================================================
        
# =============================================================================
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
# =============================================================================
        
        # 출력 
        self.clf_from_joblib = joblib.load(self.path_2ndModel) 
        self.df실거래가 = pd.DataFrame(self.openFile(self.path_실거래가))      
                
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
        
        # 아파트 별 거리 구함 
        apt_datas= []
        for i, apt_data in enumerate(self.json_apt_datas):
            
            for idx, d in enumerate(self.select_query): 
              
                ret = []

                itemLimitKm = d[1]
                
                ret = self.getInRangeSelectData(apt_data, self.items[idx], itemLimitKm)
                apt_data[self.itemKeys[idx]] = ret                
                apt_data[self.itemLenKeys[idx]] = len(ret)
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
        min_max_scaler = MinMaxScaler()
        df_matrix_MinMax = min_max_scaler.fit_transform(df_matrix)
        
        if self.simirarity == '1':
            print('*************************************************')
            print('*      유사도 비교 방법 : 코사인 유사도 비교       *')
            print('*************************************************')
            #################################
            # 코사인 유사도 
            #################################
            ret_simularity = df_matrix_MinMax
        else:
            #################################
            # Matrix Factrazation 
            ##################################
            print('*************************************************')
            print('Matrix Factrazation 비교 ')
            print('*************************************************')
            factorizer = MatrixFactorization(df_matrix_MinMax, k=6, learning_rate=0.007, reg_param=0.01, epochs=100, verbose=True)
            factorizer.fit()
            
            ret_simularity = factorizer.get_complete_matrix()
  
        
        sel_query = np.array([np.array(self.select_query).T[0]])
# =============================================================================
#         print('selq : ', sel_query)
# =============================================================================
        cosine_sim = cosine_similarity(sel_query, df.values).flatten()
        
        sim_rank_idx = cosine_sim.argsort()[::-1]
        
# =============================================================================
#         for idx in sim_rank_idx[:10]:
#             print("==== : ", self.json_apt_datas[idx])
# =============================================================================
    
        output = []
        
        first_rec_max_score = 100
        
        predict2nd = []
        
        rnkcnt = 30

        dfcomment = pd.DataFrame(self.openFile(self.path_댓글데이터))
        dfcomment['total'] = dfcomment['content']
# =============================================================================
#         dfcomment['title'] + ' ' + 
# =============================================================================
#        + dfcomment['content']  + ' ' 
        
        testCommentKey = ['월계풍림아이원','월계그랑빌','공릉풍림아이원','월계삼창']
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
        
        # p1 rank1 계산 
        for idx in sim_rank_idx[:rnkcnt]:
           
            jsonApt = self.json_apt_datas[idx]
            
# =============================================================================
#             3차 추천용 데이터 뽑기 
# =============================================================================
            aptName = jsonApt['kaptName']
            
            kAptNameSplit = aptName.split()
            
            print("========================================== : ", jsonApt['kaptName'])
          
            retDatas = []
            
            isEquals = False
            
            score_p3 = {}
            score_p3[0] = 0
            score_p3[1] = 0
            score_p3[2] = 0
            
            if len(jsonApt['doroJuso'].replace(' ','')) <= 0:
                continue
        
# =============================================================================
#             juso = jsonApt['doroJuso'].replace('서울특별시 ', '')
# =============================================================================
            #.replace(self.guName,'')            
# =============================================================================
#             print(juso);
# =============================================================================
            # 도로명 주소가 같은 애의 실거래가를 가져와서 
# =============================================================================
#             juso='동일로173가길'
# =============================================================================
            
            # 실거래가 금액 뽑아보자 
            juso_split = jsonApt['doroJuso'].split()
            
            
            print('주소 :', jsonApt['doroJuso'])
            print('주소 split 결과 :', juso_split)
           
# =============================================================================
#             (self.df실거래가['년'] == '2019') &
# =============================================================================
            dfAptInfo = self.df실거래가[
                    (self.df실거래가['도로명'] == juso_split[2]) &
                    (self.df실거래가['도로명건물본번호코드'] == juso_split[3].zfill(5)) &
                    (self.df실거래가['거래금액'].astype(int) <= int(self.myprice))
                    ]
            
            if len(dfAptInfo) <= 0:
                # 아파트 정보가 없다면 pass
                print('아파트 정보 없음 ')
                continue
                        
            maxPrice = str(dfAptInfo['거래금액'].max())
            minPrice = str(dfAptInfo['거래금액'].min())
            
            jsonApt['max_price'] = maxPrice
            jsonApt['min_price'] = minPrice
        
            print('*************************************************')
            print('*                     2차 추천                   *')
            print('*************************************************')
            
# =============================================================================
#             '월계로45가길', '94'
#             1087000000
#             & (self.df실거래가['거래금액'].astype(int) <= self.myprice))
# =============================================================================
                
            predict = 0
            
            maemae_jisu = []
            
            testkey = ['거래금액','계약년월','기준금리', '통화금융지표','매매가격지수']
            dfAptInfo2 = dfAptInfo[testkey]
            predict = self.clf_from_joblib.predict(dfAptInfo2)
            print(predict[0])
            print(predict[1])
            print(len(predict))
            print(len(dfAptInfo2))
            maemae_jisu_lastItem = {}
            
            index = 0
            for i, d in dfAptInfo2.iterrows():                  
                 jisuItem = {}
                 jisuItem['date'] = str(int(d['계약년월']))
                 jisuItem['jisu'] = str(d['매매가격지수'])
                 jisuItem['realprice'] = str(d['거래금액'])

                 jisuItem['predict_jisu'] = str(predict[index])
                 maemae_jisu.append(jisuItem);
                 maemae_jisu_lastItem = d
                 index += 1
                                
            maemae_jisu = sorted(maemae_jisu, key=(lambda maemae_jisu:maemae_jisu['date']), reverse = False)   
            

# =============================================================================
#                     if not maemae_jisu_lastItem:
# =============================================================================
            maemae_jisu_lastItem = maemae_jisu[len(maemae_jisu) - 1]
    
            jsonApt['recent_price'] = maemae_jisu_lastItem['realprice']
    
            yyyyMM = maemae_jisu_lastItem['date']
            print(yyyyMM)
            last2NdItem = dfAptInfo[dfAptInfo['계약년월'].astype(str).str.contains(yyyyMM)]
            print("***  ", last2NdItem.iloc[0])
            
            from dateutil.relativedelta import relativedelta
                               
            def createPredict2NdDataAfterMonth(lastItem, afterMonth):
                 maemaejisu = lastItem.copy()
                 nextMonth = (datetime.strptime(str(int(maemaejisu['계약년월'])), '%Y%m')+ relativedelta(months=afterMonth)).strftime("%Y%m")
                 maemaejisu['계약년월'] = nextMonth
                 predict = str(self.clf_from_joblib.predict([maemaejisu])[0])

                 ret = {}
                 ret['date'] = nextMonth
                 ret['predict_jisu'] = predict
                 
                 return ret
            
            # 1개월 뒤만 표기 
# =============================================================================
#             predict2ndAfter1MonthData = createPredict2NdDataAfterMonth(last2NdItem.iloc[0], 1)
#             
#             print('*************************************************')
#             print('*           2차 매개가격지수 1달뒤 예측               ', predict2ndAfter1MonthData['predict_jisu'])
#             print('*************************************************')
#             maemae_jisu.append(predict2ndAfter1MonthData)
#             
# =============================================================================
# =============================================================================
#                     for i in range(1,13):
#                         maemae_jisu.append(createPredict2NdDataAfterMonth(last2NdItem.iloc[0], i))
# # =============================================================================
# =============================================================================
#                     maemae_jisu.append(createPredict2NdDataAfterMonth(last2NdItem.iloc[0], 2))
#                     maemae_jisu.append(createPredict2NdDataAfterMonth(last2NdItem.iloc[0], 3))
#                     
# =============================================================================

            jsonApt['maemae_jisu'] = maemae_jisu   
                    
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
    # =============================================================================
    #                         emotion['title'] = d['title']
    # =============================================================================
                            emotion['content'] = d['content']
    # =============================================================================
    #                         emotion['text_comment'] = d['text_comment']
    # =============================================================================
                           # print(d['text_comment'])
                            label = multi_nb.predict([d['content']])[0]
                            emotion['predict'] = str(label)
                            
    # =============================================================================
    #                         if int(emotion['predict']) == 0:    
    #                             print('====== predidct 부정!!!!!!!!!!!!!!1')
    # =============================================================================
    #                        label = np.argmax(self.test_sentences([d['total']]))
    #                        emotion['label'] =  str(label)
                            
                            score_p3[label] = score_p3[label] + 1
                                                    
                            retDatas.append(emotion)
                            isEquals = False
    
                else :
                    for i,d in dfcomments.iterrows():
                        
                        emotion = {}
    # =============================================================================
    #                     emotion['title'] = d['title']
    # =============================================================================
                        emotion['content'] = d['content']
    # =============================================================================
    #                     emotion['text_comment'] = d['text_comment']
    # =============================================================================
                        
                        label = multi_nb.predict([d['content']])[0]
                        emotion['predict'] = str(label)
    # =============================================================================
    #                     label = np.argmax(self.test_sentences([d['total']]))
    #                     emotion['label'] =  str(label)
    #                     
    # =============================================================================
                        score_p3[label] = score_p3[label] + 1
                        retDatas.append(emotion)

            max_key = max(score_p3, key=score_p3.get)
            
            if max_key == 2:
                max_score_p3 = 10
                
            elif max_key == 1:
                max_score_p3 = 5
                
            else:
                max_score_p3 = 0
                
            print(score_p3)
            print(max_score_p3)
            

            
            jsonApt['comment'] = retDatas
            # score 기본값 할당 
            jsonApt['score'] = 0
            
            jsonApt['score_p1'] = str(first_rec_max_score)
            jsonApt['score_p3'] = str(max_score_p3)
            first_rec_max_score -=5
            
            output.append(jsonApt)
            
            # 아이템이 10개 이상 채워지면 종료
            if len(output) > 10:
                break

        print('* 2nd predict? : ',int(self.optSecond))
        
        if int(self.optSecond) == 1:
                
            # p2 rank 계산
            min_max_scaler = MinMaxScaler()
            p2 = np.array(predict2nd)
            p2 = p2.reshape(rnkcnt,1)
         
            predictMinMax = min_max_scaler.fit_transform(p2)
            print('predict score = ', predictMinMax * 40)
            p2Scores = predictMinMax * 40
            
            finalScore = []
            for i, d in enumerate(output):
                p2score = p2Scores[i][0]
# =============================================================================
#                 d['score_p2'] = str(int(p2score))
# =============================================================================
                d['score_p2'] = str(int(0))
                d['score'] = float(d['score_p1']) + float(p2score) + float(d['score_p3'])
                
                score_sim = (float(d['score']) /150) * 100
                d['score_sim'] = str(round(score_sim,2))
                
                finalScore.append(d)
            
            result = sorted(finalScore, key=(lambda finalScore:finalScore['score']), reverse = True)        
            return result
        
        else:
            
            result =[]
            for d in output:
                d['score_p2'] = 0
                d['score_p3'] = 0
                d['score'] = float(d['score_p1'])
                
                score_sim = (float(d['score']) /100) * 100
                d['score_sim'] = str(round(score_sim,2))
                result.append(d)
            
            return result
    

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
                            inRangeItems.append(item)

            except:
# =============================================================================
#                 print(item)
# =============================================================================
                pass
                 


        return inRangeItems
    
        
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
        for epoch in range(self._epochs):

            # rating이 존재하는 index를 기준으로 training
            for i in range(self._num_users):
                for j in range(self._num_items):
                    if self._R[i, j] > 0:
                        self.gradient_descent(i, j, self._R[i, j])
            cost = self.cost()
            self._training_process.append((epoch, cost))

            # print status
            if self._verbose == True and ((epoch + 1) % 10 == 0):
                print("Iteration: %d ; cost = %.4f" % (epoch + 1, cost))


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
    
    print('1 : ', d['select1'])
    print('2 : ', d['select2'])
    print('3 : ', d['select3'])
    print('4 : ', d['select4'])
    print('5 : ', d['select5'])
    print('6 : ', d['select6'])
    print('7 : ', d['select7'])
    print('8 : ', d['select8'])
    print('9 : ', d['select9'])
    print('myprice : ', d['myprice'])
    
    recommend.setup("1", d['gu'], d['optSecond'], d['select1'], d['select2'],d['select3'],d['select4'],d['select5'],d['select6'],d['select7'],d['select8'],d['select9'],d['myprice'])
    out = recommend.run(multi_nbc)
    
# =============================================================================
#     for o in out:
#         print(o['kaptName'])
#         
# =============================================================================
    data = {}
    data['data'] = out
    ojson = jsonify(data)
    
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
stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

multi_nbc = Pipeline([('vect', CountVectorizer(ngram_range=(2, 2),stop_words = stopwords)),
                      ('tfidf', TfidfTransformer()),
                      ('nbc', MultinomialNB())])


if __name__ == "__main__":
    
 
    
    df = pd.DataFrame()
    
    xlspath = 'datas/labelData.xlsx'
    df1_1 = pd.read_excel(xlspath,sheet_name='4-1.내용(2019,1~500)')
    df1_2 = pd.read_excel(xlspath,sheet_name='4-2.내용(2019,501~948)')
    
    df2_1 = df1_1[['내용','긍정-2, 중립-1, 부정-0']]
    df2_2 = df1_2[['내용','긍정-2, 중립-1, 부정-0']]
    
    df2_1 = df2_1.dropna()
    df2_2 = df2_2.dropna()
    
    df = pd.concat([df2_1, df2_2])

    train = []
    label = []
    for i, d in df.iterrows():
        # 중립 제거
        # ,(콤마) 제거 
        l = int(d['긍정-2, 중립-1, 부정-0'])
        if l != 1:        
            data = d['내용'][d['내용'].find(',') + 1:]
            train.append(data)
            label.append(int(d['긍정-2, 중립-1, 부정-0']))
    
    from sklearn.model_selection import train_test_split
    
    print('.', end = '')
    X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0.2)


    start = time()
    multi_nbc.fit(X_train, y_train)
    end = time()
#     print('Time: {:f}s'.format(end-start))
    y_pred = multi_nbc.predict(X_test)
    accScore = accuracy_score(y_test, y_pred)
    print("3차 테스트 테스트 정확도: {:.3f}".format(accScore))
            
    app.run(host='0.0.0.0',port=5001)
    
    
    
    
    
    

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
