import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st 
import datetime as dt
# from datetime import datetime
import altair as alt
import plotly.express as px


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


# storge in a database
import sqlite3
conn = sqlite3.connect('data.db')
c = conn.cursor()

# Create table
# Create function from sql
def create_table():
    c.execute('CREATE TABLE IF NOT EXISTS predictionTable(message TEXT, tokens TEXT, predicted TEXT, postdate DATE)')

def add_data(message,tokens, predicted, postdate):
    c.execute('INSERT INTO predictionTable(message, tokens, predicted, postdate) VALUES (?,?,?,?)', (message,tokens, predicted, postdate))
    conn.commit()

def view_all_data():
    c.execute('SELECT * FROM predictionTable')
    data = c.fetchall()
    return data

def main():
    menu = ["Home", "Report", "About"]
    create_table()
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Home")
        
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
                postdate = dt.datetime.now()
                add_data(message, my_tokens, my_predictions[0], postdate)
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
        stored_data = view_all_data()
        new_df = pd.DataFrame(stored_data, columns=['message', 'tokens', 'predicted', 'postdate'])
        st.dataframe(new_df)
        new_df['postdate'] = pd.to_datetime(new_df['postdate'])
        # st.write(new_df['postdate'])
        
        st.title('กราฟรายวัน')
        # get today's date
        today = dt.date.today()
        
        # filter the datafreame to show only today's data
        todays_posts = new_df[new_df['postdate'].dt.date == today]
        # filter only the predicted column        
        todays_posts = todays_posts[['predicted']]       
        
        # normal bar chart
        # counts = todays_posts['predicted'].value_counts()        
        # st.bar_chart(counts)
        
        # bar chart using plotly express
        counts2 = todays_posts['predicted'].value_counts().reset_index()
        # Define the color of the bars
        colors = {'N': 'งานอื่นๆ', 'Y': 'งานบุคคล'}
        
        # Map the colors to the predicted values
        counts2['color'] = counts2['index'].map(colors)          
        
        # Create a bar chart using Plotly Express
        fig = px.bar(counts2, x='index', y='predicted', color='color')
        
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
        
        
    

if __name__ == '__main__':
    main()


