import streamlit as st
import os
import kagglehub
from PIL import Image
# import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from agent import Agent

# st.logo('media/Logo2.png', size='large')
# st.set_page_config(page_icon='media/iconNationAi.png', page_title='Medical Agent', layout='centered')

os.environ["KAGGLE_USERNAME"] = st.secrets["KaggleUserName"]
os.environ["KAGGLE_KEY"] = st.secrets["KaggleAPIkey"]

if 'agent' not in st.session_state:
    with st.spinner('Please wait, Loading Data Sources...'):
        st.session_state.agent = Agent(st.secrets['GeminiAPI'])
if 'language' not in st.session_state:
    st.session_state.language = 'Arabic'
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'patient_report' not in st.session_state:
    st.session_state.patient_report = ""


def show_imaging(ImageID, cap):
    """btgeeb elsora mn kaggle dataset"""
    handle = "seoyunje/padchest-small-dataset"
    path_to_image = f"PC/images-224/images-224/{ImageID}"
    try:
        local_image_path = kagglehub.dataset_download(
            handle=handle, 
            path=path_to_image
        )
        img = Image.open(local_image_path)
        if img.mode in ("I;16", "I", "I;16L", "I;16B"):
            img_np = np.array(img, dtype=np.float32)
            img_normalized = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255.0).astype(np.uint8)
            img_to_display = Image.fromarray(img_normalized)
        else:
            img_to_display = img.convert("RGB")
        st.columns([1,1,1])[1].image(img_to_display, caption=cap, width='stretch')

    except Exception as e:
        st.error(f"Error while loading image!: {e}")

def show_report():
    """3ard al report lel patient"""
    report = st.session_state.patient_report
    with st.expander('Patient Report'):
        study_date = str(report['StudyDate']) 
        st.caption(f"Study Date: {study_date[-2:]}/{study_date[-4:-2]}/{study_date[:4]}", text_alignment='right')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(":blue[Patient ID:]  \n" + str(report['PatientID']))
        with col2:
            st.markdown(":blue[Study ID:] " + str(report['StudyID']))
        with col3:
            st.markdown(":blue[Report ID:]  \n" + str(report['ReportID']))
            
        st.divider()
        col1, col2, col22, col3, col4 = st.columns([2,0.6,0.4,1,1])
        with col1:
            st.markdown("Name  \n" + report['Name'])
        with col2:
            st.markdown("Gender  \n" + report['PatientSex'])
        with col22:
            st.markdown(f"Bed  \n{report['Bed']}")
        with col3:
            st.markdown(f"Age in the Study  \n{report['Age']} year")
        with col4:
            st.markdown(f"Current Age  \n{datetime.now().year-int(report['PatientBirth'])} year")
        st.space('small')
        st.markdown(f":blue[Diagnosis:] {str(report['Labels'])[1:-1]}")
        
        st.markdown(":blue[Report Content:]  \n" + report['Report'])
        study_date = str(report['StudyDate'])
        caption = f"X-Ray Image: {study_date[-2:]}/{study_date[-4:-2]}/{study_date[:4]}  \nView Position: {report['ViewPosition_DICOM']}"

        show_imaging(report['ImageID'], caption)

if st.session_state.messages:
    for message in st.session_state.messages:
        if message[0]=="human":
            st.chat_message("user", avatar='media/user avatar.png').markdown(message[-1], text_alignment="right" if st.session_state.language =='Arabic' else 'left')
        elif message[0]=="ai":
            st.chat_message("assistant", avatar='media/bot avatar.png').markdown(message[-1], text_alignment="right" if st.session_state.language =='Arabic' else 'left')
    mt = st.empty()
else:
    greeting_en = "Hello 👋, I am , how can i help you today?"
    greeting_ar = "أهلا بيك 👋, انا, أقدر اساعدك ازاي النهاردة؟"
    st.chat_message("assistant", avatar='media/bot avatar.png').markdown(greeting_ar if st.session_state.language=='Arabic' else greeting_en, 
                                                                         text_alignment='right' if st.session_state.language =='Arabic' else 'left')
    mt = st.empty()
    
message = st.chat_input("Say Something...", accept_audio=True)

if message:
    # if message.files:
    #     pass
    with mt.chat_message("assistant", avatar='media/bot avatar.png').spinner():
        if message.text:
            st.session_state.messages, report = st.session_state.agent.predict(message.text)
            if report:
                st.session_state.patient_report = report
            st.rerun()
        if message.audio:
            with open("user_voice.mp3", "wb") as f:
                f.write(message.audio.read())
            st.session_state.messages, report = st.session_state.agent.predict("user_voice.mp3", voice=True, language=st.session_state.language)
            if report:
                st.session_state.patient_report = report
            st.rerun()
       

if st.session_state.patient_report:
    show_report()

# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# if 'current_chat' not in st.session_state:
#     st.session_state.current_chat = 0
# if 'report_history' not in st.session_state:
#     st.session_state.report_history = []

# with st.sidebar:
#     st.header('Conversation History', divider='blue')
#     col1, col2 = st.columns([8,1])
#     with col1:
#         if st.columns([1,4,1])[1].button(":material/add: New chat", type='tertiary', width='stretch'):
#             if st.session_state.messages not in st.session_state.chat_history:
#                 st.session_state.chat_history.append(st.session_state.messages)
#                 st.session_state.report_history.append(st.session_state.patient_report)
#             st.session_state.messages = []
#             st.session_state.patient_report = ""
#             st.session_state.current_chat = len(st.session_state.chat_history)
#             st.rerun()

#     if st.session_state.chat_history:
#         col1, col2 = st.columns([8,1])
#         k = len(st.session_state.chat_history)
#         for h in range(len(st.session_state.chat_history)-1, -1, -1):
#             with col1:
#                 if st.button(f'chat {k}', width='stretch', type='tertiary'):
#                     if st.session_state.current_chat >= len(st.session_state.chat_history):
#                         st.session_state.chat_history.append(st.session_state.messages)
#                         st.session_state.report_history.append(st.session_state.patient_report)
#                     else:
#                         st.session_state.chat_history[st.session_state.current_chat] = st.session_state.messages
#                         st.session_state.report_history[st.session_state.current_chat] = st.session_state.patient_report

#                     st.session_state.messages = st.session_state.chat_history[h]
#                     st.session_state.current_chat = h
#                     st.rerun()
#             with col2:
#                 if st.button(':material/delete:', key=k, type='tertiary'):
#                     st.session_state.chat_history.remove(st.session_state.chat_history[h])
#                     st.rerun()
#             k-=1
        
    



# Settings
with st.sidebar:
    # st.divider()
    st.header(":material/settings: Settings")
    new_lang = st.selectbox(f"**Voice Language** (current is {st.session_state.language})", ('Arabic', 'English'), placeholder='Language..', index=None)
    if new_lang and new_lang != st.session_state.language:
        st.write(f"Language will be change into :blue[{new_lang}]")
        if st.button("Save changes", type='primary'):
            st.session_state.language = new_lang
            st.rerun()

