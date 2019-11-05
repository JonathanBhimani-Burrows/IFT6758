import pandas as pd
import pickle

class relations_agglomerator():
    def __init__(self):
        self.data = pd.DataFrame(
            columns=['like_id', 'age', 'gender', 'ope', 'con', 'ext', 'agr', 'neu'])
        
        self.confidence = pd.DataFrame(
            columns=['like_id', 'age', 'gender'])
    
    def train(self, relation_data, profile_data):
        total_pages = len(relation_data['like_id'].unique())
        grouped_relations = relation_data.groupby('like_id')
        i = 0

        for _, page in grouped_relations:
            page_traits = pd.merge(page, profile_data, on=['userid'], how='left')
            page_traits = page_traits[['like_id', 'age', 'gender', 'ope', 'con', 'ext', 'agr', 'neu']]
            
            self.data.loc[len(self.data)] = ([
                page_traits.iloc[0]['like_id'],
                page_traits['age'].value_counts().idxmax(),
                page_traits['gender'].value_counts().idxmax(),
                page_traits['ope'].mean(),
                page_traits['con'].mean(),
                page_traits['ext'].mean(),
                page_traits['agr'].mean(),
                page_traits['neu'].mean()
            ])

            self.confidence.loc[len(self.confidence)] = ([
                page_traits.iloc[0]['like_id'],
                page_traits['age'].value_counts().max() / page_traits['age'].value_counts().sum(),
                page_traits['gender'].value_counts().max() / page_traits['gender'].value_counts().sum()
            ])

            i += 1

            if i % 1000 == 0:
                print("Completed agglomeration for % 6d pages (% 3.1f%%)" %(i * 100, i / total_pages))

        print('Training complete for relation agglomerator')

    def predict(self, relation_data, uid):
        relation_data = relation_data[relation_data['userid'] == uid]

        liked_page_data = pd.merge(relation_data, self.data, on=['like_id'], how='left')

        if liked_page_data['age'].isnull().all() == False:
            prediction = [liked_page_data['age'].value_counts().idxmax(),
                          int(liked_page_data['gender'].value_counts().idxmax()),
                          liked_page_data['ope'].mean(),
                          liked_page_data['con'].mean(),
                          liked_page_data['ext'].mean(),
                          liked_page_data['agr'].mean(),
                          liked_page_data['neu'].mean()]

            return prediction
        
        else:
            return -1
    