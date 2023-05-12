import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st 
import datetime as dt
from datetime import datetime, timedelta
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

# @st.cache_data
def get_data_from_csv():
    df = pd.read_csv('BinaryClass@DB.csv')
    return df


st.set_page_config(page_title="วิเคราะห์ข้อความเพื่อจำแนกงาน", page_icon=":writing_hand:")

df = get_data_from_csv()


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


# Timezone
import pytz
tz = pytz.timezone('Asia/Bangkok')

@st.cache_resource
def get_db():
    key_dict = json.loads(st.secrets["textkey"])
    creds = service_account.Credentials.from_service_account_info(key_dict)
    db = firestore.Client(credentials=creds, project="text-classified-jobs")
    return db

def post_message(db: Client, message, tokens, predicted):
    #-- cali
    payload = {
        "message": message,
        "tokens": tokens,
        "predicted": predicted,
        "date": datetime.now(tz).strftime("%Y/%m/%d %H:%M:%S"),
        "user": name,
    }
    doc_ref = db.collection("jobclassifier2").document()

    doc_ref.set(payload)
    return


# ==Display report
def all_data(name):
    
    db = get_db()
                
    posts = list(db.collection(u'jobclassifier2').stream())
    posts_dict = list(map(lambda x: x.to_dict(), posts))
    df = pd.DataFrame(posts_dict)
                    
    new_df = pd.DataFrame(df, columns=[ 'date','message','predicted','user'])
            
      
            
    # Convert Timestamp column to datetime format
    new_df['date'] = pd.to_datetime(new_df['date'], errors='coerce')
    new_df['date'] = pd.to_datetime(new_df['date'])
    
    user_posts = new_df[new_df['user'] == name]
            
    st.dataframe(user_posts)
    
    





def today_report(name):
    # st.write(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
    db = get_db()
                
    posts = list(db.collection(u'jobclassifier2').stream())
    posts_dict = list(map(lambda x: x.to_dict(), posts))
    df = pd.DataFrame(posts_dict)
                    
    new_df = pd.DataFrame(df, columns=[ 'date','message','predicted','user'])
            
    # display all data
    # st.dataframe(new_df)
            
    # Convert Timestamp column to datetime format
    new_df['date'] = pd.to_datetime(new_df['date'])
            
    # Filter DataFrame to only include rows with timestamps matching today's date
    today = datetime.today().strftime('%Y/%m/%d')
    todays_posts = new_df[new_df['date'].dt.strftime('%Y/%m/%d') == today]
    
    # filter only the logged in user
    user_posts = todays_posts[todays_posts['user'] == name]
    # st.write(user_posts)     
    
    st.subheader('งานที่คุณทำวันนี้')
    
                            
    # filter only the predicted column        
    user_post_predicted = user_posts[['predicted']]       
    # st.dataframe(todays_posts)
    # normal bar chart
    # counts = todays_posts['predicted'].value_counts()        
    # st.bar_chart(counts)
            
    # bar chart using plotly express
    dailycount = user_post_predicted['predicted'].value_counts().reset_index()
            
    # st.write(dailycount)
            
    # Define the color of the bars
    colors = { 'Y': 'งานบุคคล', 'N': 'งานอื่นๆ'}
    # colors2 = ['#00A300', '#FF6961']
            
    # Map the colors to the predicted values
    dailycount['color'] = dailycount['index'].map(colors)          
            
    # # Create a bar chart using Plotly Express
    fig = px.bar(dailycount, x='index', y='predicted', color='color', color_discrete_map={'งานบุคคล': '#00A300', 'งานอื่นๆ': '#FF6961'})
            
    st.plotly_chart(fig)
    
    # รายละเอียดงานที่ทำวันนี้
    st.subheader("รายละเอียดงานที่ทำวันนี้")
    # st.dataframe(todays_posts[['message','predicted','date']])
    
    
                    
    st.subheader("งานฝ่ายบุคลากร")
    # st.write(user_posts)
        
    # filter only the predicted column is Y
    yes = user_posts[user_posts['predicted'] == 'Y']
    # st.dataframe(yes)
    
    df_yes = pd.DataFrame(yes[['message','predicted','date']])
    # # CSS to inject contained in a string
    hide_table_row_index = """
        <style>
        thead tr th:first-child {display:none}
        tbody th {display:none}
        </style>
        """

    # # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    st.table(df_yes)

    
    st.subheader("งานอื่นๆ")
    # filter only the predicted column is N
    no = user_posts[user_posts['predicted'] == 'N']
    
    df_no = pd.DataFrame(no[['message','predicted','date']])
    
    
    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    st.table(df_no)
 
 
def weekly_report(name):
    
    db = get_db()
                
    posts = list(db.collection(u'jobclassifier2').stream())
    posts_dict = list(map(lambda x: x.to_dict(), posts))
    df = pd.DataFrame(posts_dict)
                    
    new_df = pd.DataFrame(df, columns=[ 'date','message','predicted','user'])
    # st.write(new_df)
    
    # filter only the logged in user
    user_post_week = new_df[new_df['user'] == name]
    # st.write(user_post_week) 
       
            
    # Convert Timestamp column to datetime format
    new_df['date'] = pd.to_datetime(user_post_week['date'])
              
    # extract day of the week
    new_df['day_of_week'] = new_df['date'].dt.weekday
    
    
    # filter for workdays (Monday to Friday)
    df_workdays = new_df[new_df['day_of_week'] < 5]
    # or df_workdays = new_df[new_df['day_of_week'].isin([0,1,2,3,4])]
    
    # display the data for workdays only
    st.write(df_workdays)
         
            


    



# for authentification system
import pickle
from pathlib import Path
import streamlit_authenticator as stauth

# --- Authentication ---
names = ["โทนี่ สตาร์ค","ronnachai.th","Supachai","Jutamas","Naphatsamon","Yaowalak","Phalawan","Thanchanathorn","Chanapon","Rattikan","Manatsanan","Yonlada","Narupong","Chaiwat","Tanakrisana","Nalinthorn","Phithak","Nateesut","Krittayot"]
usernames = ["ironman","ronnachai.th","supachai","jutamas","naphatsamon","yaowalak","phalawan","thanchanathorn","chanapon","rattikan","manatsanan","yonlada","narupong","chaiwat","tanakrisana","nalinthorn","phithak","nateesut","krittayot"]

# Load hashed passwords from file
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
    "sales_dashboard", "abcdef", cookie_expiry_days=30)

name, authentication_status,usernames = authenticator.login("เข้าสู่ระบบวิเคราะห์ข้อความเพื่อจำแนกงาน","main")

if authentication_status == False:
    st.error("คุณไม่ได้รับอนุญาตให้เข้าถึงหน้านี้")

if authentication_status == None:
    st.warning("กรุณาเข้าสู่ระบบ")

if authentication_status == True:
    # st.success("ยินดีต้อนรับคุณ {}".format(name))
    
    # --- sidebar ---
    authenticator.logout("ออกจากระบบ","sidebar")
    st.sidebar.title(f"ยินดีต้อนรับคุณ {name}")
    







# --- for authentification system

    def main():
        menu = ["หน้าหลัก", "รายงานสรุป", "เกี่ยวกับ"]
        
        # # ---- Hide streamlit menu ----
        # hide_streamlit_style = """
        # <style>
        # #MainMenu {visibility: hidden;}
        # footer {visibility: hidden;}
        # header {visibility: hidden;}
        # </style>
        # """
        # st.markdown(hide_streamlit_style, unsafe_allow_html=True)
        
        
        
        choice = st.sidebar.selectbox("Menu", menu)
        
        st.success("ยินดีต้อนรับคุณ {}".format(name))
        
        if choice == "หน้าหลัก":
            st.subheader("หน้าหลัก")
            
            db = get_db()
            
            
            with st.form(key='mlform', clear_on_submit=True):
                col1, col2 = st.columns([2,1])
                with col1:                
                    message = st.text_input("บันทึกงานที่ได้รับมอบหมาย","")
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
            today_report(name)
                            
                            
                            
                                    
        elif choice == "รายงานสรุป":
            
            
            st.subheader("รายงานสรุป")  
            
            
            
            with st.expander("ดูข้อมูลทั้งหมด"):
                all_data(name)
                
        
            st.subheader('เลือกดูตามช่วงวันที่กำหนด')
            today = datetime.now(tz).strftime("%Y/%m/%d")    
            today = datetime.strptime(today, "%Y/%m/%d")
            # today = datetime.date.today()
            tomorrow = today + dt.timedelta(days=1)
            start_date = st.date_input('วันที่เริ่มต้น', today)
            end_date = st.date_input('วันที่สิ้นสุด', tomorrow)
            if start_date < end_date:
                st.success('วันที่เริ่มต้น: `%s`\n\nวันที่สิ้นสุด:`%s`' % (start_date, end_date))
                db = get_db()
                
                posts = list(db.collection(u'jobclassifier2').stream())
                posts_dict = list(map(lambda x: x.to_dict(), posts))
                df = pd.DataFrame(posts_dict)
                    
                new_df = pd.DataFrame(df, columns=[ 'date','message','predicted','user'])
    
                user_posts = new_df[new_df['user'] == name]
            
                
            
                # convert the date column to datetime
                user_posts['date'] = pd.to_datetime(user_posts['date']).dt.date
          
                # Query the posts between two dates            
                filtered_df = user_posts[(user_posts['date'] >= start_date) & (user_posts['date'] <= end_date)]
                # Print the filtered DataFrame
                st.write(filtered_df)
                
                gday = filtered_df.groupby(['date','predicted'])[['message']].count().reset_index()
                st.write(gday)
                
                group_data = filtered_df.groupby(['date','predicted'])
                # group_data=group_data.sum()
                st.dataframe(group_data.sum())
                
                
           
            else:
                st.error('ข้อผิดพลาด: วันที่สิ้นสุดต้องอยู่หลังวันที่เริ่มต้น')
            
            
            
            
                
                
                
                
                
                
                
            st.write('กราฟรายสัปดาห์ coming soon')
            
            st.write('กราฟรายเดือน coming soon')
            
            


        else:
            st.subheader("เกี่ยวกับ")
            st.write('This app is building by hard-working researchers team.')
            st.write("Line กลุ่ม เพื่อข้อเสนอแนะนำ")
            st.markdown('<a href="https://forms.gle/yFGum9EoXrsiJX948" target="_blank">ข้อเสนอแนะ Google Form</a>', unsafe_allow_html=True)
            st.markdown('<a href="https://app.sli.do/event/9aFBBjm1RBDUo1Q5MYoYPn/live/questions" target="_blank">ถามและตอบ, Q&A, ใช้งานได้วันจันทร์ที่ 24 ถึงศุกร์ที่ 28 เมษายน 2566</a>', unsafe_allow_html=True)
            todayAbout = datetime.now(tz).strftime("%Y/%m/%d %H:%M:%S")
            st.write(f"วันนี้ : {todayAbout}")
                
            
        

    if __name__ == '__main__':        
        main()


