import streamlit as st

st.set_page_config(page_title="Medical Agent", page_icon='media/iconNationAi.png', initial_sidebar_state='collapsed', layout='centered')
st.logo('media/Logo2.png', size='large')


home = st.Page("home.py", title="Home", icon=":material/home:", default=True)
chat = st.Page("st_agent.py", title="Chatbot", icon=":material/smart_toy:")

pg = st.navigation([home, chat])

pg.run()
