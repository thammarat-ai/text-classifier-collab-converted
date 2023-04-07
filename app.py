import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st 


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
# def create_table():
#     c.execute('CREATE TABLE IF NOT EXISTS predictionTable(message TEXT, prediction TEXT, probability NUMBER, software_proba NUMBER, hardware_proba NUMBER, postdate DATE)')

# def add_data(message,prediction, probability, software_proba, hardware_proba, postdate):
#     c.execute('INSERT INTO predictionTable(message, prediction, probability, software_proba, hardware_proba, postdate) VALUES (?,?,?,?,?,?)', (message,prediction, probability, software_proba, hardware_proba, postdate))
#     conn.commit()

# def view_all_data():
#     c.execute('SELECT * FROM predictionTable')
#     data = c.fetchall()
#     return data

def main():
    menu = ["Home", "Report", "About"]
    #create_table()
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Home":
        st.subheader("Home")
        
        with st.form(key='mlform'):
            col1, col2 = st.columns([2,1])
            with col1:                
                message = st.text_area("บันทึกงานที่ได้รับมอบหมาย", "กรอกงานที่ทำ เช่น การจัดทำแผนพัฒนาบุคลากร", height=200)
                submit_message = st.form_submit_button(label='บันทึกงาน')
            with col2:
                st.write("AI ช่วยวิเคราะห์งานที่ทำเป็นงานงานฝ่ายบุคลากร")
                st.write("จะทำนายว่าเป็นงานฝ่ายบุคคลหรือ อื่นๆ")
                
        if submit_message:
            my_tokens = text_process(message)
            my_bow = cvec.transform(pd.Series([my_tokens]))
            my_predictions = lr.predict(my_bow)
            
            #postdate = datetime.datetime.now()
            # add data to database
            # call function add_data
            
            
            
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
        st.write('กราฟรายวัน')
        st.write('กราฟรายสัปดาห์')
        st.write('กราฟรายเดือน')
        st.write('กราฟรายปี')
        
    else:
        st.subheader("About")
        st.write('This app is built by gig')
    

if __name__ == '__main__':
    main()


