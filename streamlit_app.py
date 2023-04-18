import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st 
import datetime as dt
from datetime import datetime
import altair as alt
import plotly.express as px
import json


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import model_selection, preprocessing, metrics

from pythainlp.corpus.common import thai_stopwords
from pythainlp import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report

from sklearn.linear_model import LogisticRegression


df = pd.read_csv('BinaryClass@DB.csv')

#set thai_stopwords
from pythainlp.corpus.common import thai_stopwords
thai_stopwords = list(thai_stopwords())

#cleansing unused
def text_process(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", '"', "ๆ", "ฯ"))
    final = word_tokenize(final)
    final = " ".join(word for word in final)
    final = " ".join(word for word in final.split() 
                     if word.lower not in thai_stopwords)
    return final
    
df['Text_tokens'] = df['Text'].apply(text_process)

# set split validation data = train(70%) and test(30%) 
X = df[['Text_tokens']]
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# feature_extraction using CountVector
cvec = CountVectorizer(analyzer=lambda x:x.split(' '))
cvec.fit_transform(X_train['Text_tokens'])
# st.write(cvec.vocabulary_)

#transform CountVector to feature
train_bow = cvec.transform(X_train['Text_tokens'])
pd.DataFrame(train_bow.toarray(), columns=cvec.get_feature_names_out(), index=X_train['Text_tokens'])

# st.write(pd.DataFrame(train_bow.toarray(), columns=cvec.get_feature_names_out(), index=X_train['Text_tokens']))

lr = LogisticRegression() #***the best **
lr.fit(train_bow, y_train)

test_bow = cvec.transform(X_test['Text_tokens'])
test_predictions = lr.predict(test_bow)

# storge in firestore
import json

from google.cloud import firestore
from google.cloud.firestore import Client
from google.oauth2 import service_account

@st.cache_resource
def get_db():
    key_dict = json.loads(st.secrets["textkey"])
    creds = service_account.Credentials.from_service_account_info(key_dict)
    db = firestore.Client(credentials=creds, project="text-classified-jobs")
    return db

def post_message(db: Client, message, tokens, predicted):
    payload = {
        "message": message,
        "tokens": tokens,
        "predicted": predicted,
        "date": datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
    }
    doc_ref = db.collection("jobclassifier").document()

    doc_ref.set(payload)
    return


def main():
    menu = ["Home", "Report", "About"]
    # create_table()
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Home")
        db = get_db()
        
        
        with st.form(key='mlform', clear_on_submit=True):
            col1, col2 = st.columns([2,1])
            with col1:                
                message = st.text_area("บันทึกงานที่ได้รับมอบหมาย", "", height=200)
                submit_message = st.form_submit_button(label='บันทึกงาน')
            with col2:
                st.write("AI ช่วยวิเคราะห์งานที่ทำเป็นงานงานฝ่ายบุคลากร")
                st.write("จะทำนายว่าเป็นงานฝ่ายบุคคลหรือ อื่นๆ")
                
            if submit_message:
                
                if message == "":
                    st.warning("กรุณากรอกงานที่ได้รับมอบหมายก่อน")
                    st.stop()           
                else:
                    my_tokens = text_process(message)
                    my_bow = cvec.transform(pd.Series([my_tokens]))
                    my_predictions = lr.predict(my_bow)
                    
                    # add data to database
                    # call function add_data
                    post_message(db, message, my_tokens, my_predictions[0])
                    
                    st.info("ข้อความที่ทำการวิเคราะห์")
                    st.write(message)
                            
                    st.success("ผลการวิเคราะห์")
                    if my_predictions[0] == 'Y':
                        st.write('งานฝ่ายบุคคล')
                        st.success("วิเคราะห์งานเรียบร้อย")
                    else:
                        st.write('งานอื่นๆ')
                        st.warning("วิเคราะห์งานเรียบร้อย")   
                                   
    elif choice == "Report":
        st.subheader("Report")
        db = get_db() 
               
        posts = list(db.collection(u'jobclassifier').stream())
        posts_dict = list(map(lambda x: x.to_dict(), posts))
        df = pd.DataFrame(posts_dict)
                
        new_df = pd.DataFrame(df, columns=[ 'date','message','predicted'])
        
        st.dataframe(new_df)
        
        # Convert Timestamp column to datetime format
        new_df['date'] = pd.to_datetime(new_df['date'])
        
        # Filter DataFrame to only include rows with timestamps matching today's date
        today = datetime.today().strftime('%Y/%m/%d')
        todays_posts = new_df[new_df['date'].dt.strftime('%Y/%m/%d') == today]
        
        st.write(todays_posts)
        
        st.title('กราฟรายวัน')
                
        
        # filter only the predicted column        
        todays_posts = todays_posts[['predicted']]       
        # st.dataframe(todays_posts)
        # normal bar chart
        # counts = todays_posts['predicted'].value_counts()        
        # st.bar_chart(counts)
        
        # bar chart using plotly express
        counts2 = todays_posts['predicted'].value_counts().reset_index()
        st.write(counts2)
        # Define the color of the bars
        colors = { 'Y': 'งานบุคคล', 'N': 'งานอื่นๆ'}
        # colors2 = ['#00A300', '#FF6961']
        
        # Map the colors to the predicted values
        counts2['color'] = counts2['index'].map(colors)          
        
        # # Create a bar chart using Plotly Express
        fig = px.bar(counts2, x='index', y='predicted', color='color', color_discrete_map={'งานบุคคล': '#00A300', 'งานอื่นๆ': '#FF6961'})
        
        st.plotly_chart(fig)
        
        

        
        
        
        st.write('กราฟรายสัปดาห์')
        # filter the dataframe to show only the posts made on workdays
        # workday_posts = new_df[new_df['postdate'].dt.weekday.between(0, 4)]
        # st.write(workday_posts)
        # counts2 = workday_posts['predicted'].value_counts()
        # st.bar_chart(counts2)
        

        st.write('กราฟรายเดือน')
        # filter the dataframe to show only the posts made in a particular month
        # target_month = 4 # for example, we want to show posts from April
        # monthly_posts = new_df[new_df['postdate'].dt.month == target_month]
        # st.write(monthly_posts)
        
        st.write('กราฟรายปี')
        # filter the dataframe to show only the posts made in a particular year
        # target_year = 2023 # for example, we want to show posts from the year 2023
        # yearly_posts = new_df[new_df['postdate'].dt.year == target_year]
        # st.write(yearly_posts)
        
    else:
        st.subheader("About")
        st.write('This app is built by gig')
        stored_data = view_all_data()
        new_df = pd.DataFrame(stored_data, columns=['message', 'tokens', 'predicted', 'postdate'])
        @st.cache_data
        def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        csv = convert_df(new_df)
        
        st.download_button(
            label="Download CSV data",
            data=csv,
            file_name='predictionTable.csv',
            mime='text/csv'
        )
        
        
        
    

if __name__ == '__main__':
    st.set_page_config(page_title="วิเคราะห์ข้อความเพื่อจำแนกงาน", page_icon="👨‍💻")
    main()


