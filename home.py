import streamlit as st
import streamlit.components.v1 as components


# st.subheader("AI Medical Agent - NationAI Assesment", divider='blue')

st.set_page_config(page_title="Medical Agent/Home", initial_sidebar_state='collapsed', layout='wide')

chat = st.Page('st_agent.py')

with open('data/html.txt','rb') as f:
    particles_js = f.read()

col1, col2, col3 = st.columns([1,4,1])

with col1:
    components.html(particles_js, height=1600, scrolling=False)
with col3:
    components.html(particles_js, height=1600, scrolling=False)
with col2:
    st.header("The Smartest AI Medical Assistant\n\n## Ready to Chat!", text_alignment='center')
    st.markdown(":rainbow[Patient's reports assistant and guidelines, medical information retrieval]", text_alignment='center')
    mt = st.empty()
    st.columns([1.47,1,1], vertical_alignment='top')[1].image(r"media/Ai Robot Vector Art.gif")
    if st.columns([2,1,2])[1].button(":material/chat: Chat", type='primary', width='stretch'):
        st.switch_page(chat)
    # st.page_link(chat, label='Chat', )
    # col1, col2, col3, col4 = st.columns([3,1.1,1.1,3])
    # with col2:
    #     st.link_button(":material/play_arrow: Demo", type='secondary', url='https://youtu.be/1qybCcar-vE?si=l1EPUSfeNxKg1uXk', width='stretch')
    # with col3:
    #     if st.button(":material/chat: Chat", type='primary', width='stretch'):
    #         st.switch_page(chat)
    st.divider()


with col2:
    st.subheader("Introduction", divider='blue', width='content')
    st.markdown("""
    **AI Medical Agent key features:**
    1. Support text and voice interaction for faster communication (Arabic & English)
    2. Support patient's reports & informations
    3. Suuport medical information about healthcare from "The Washington Manual of Medical Therapeutics" book
    4. Support 3 types of knowledge resources (Patients data frame, medical knowledge base RAG system, Kaggle api access for patient's medical imaging scan images)
    5. Guard to avoice diagnostic and out of context questions
    """)
    st.divider()
with col2:
    st.subheader(":material/settings: Project workflow", divider='blue', width='content')
    st.image("media/project flow.svg")

st.divider()
