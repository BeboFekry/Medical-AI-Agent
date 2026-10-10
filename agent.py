from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import speech_recognition as sr
import pandas as pd
from pydantic import BaseModel, Field

class MedicalGuardrail(BaseModel):
    is_diagnosis_attempt: bool = Field(
        description="True ONLY if the user is asking for a personal medical diagnosis based on their symptoms. False for general searches, patient data lookups, or medical guidelines."
    )
    is_out_of_context: bool = Field(
        description="True ONLY if the user is asking about topics completely unrelated to (medicine, patient data, medical reports, healthcare, greeting or thnaking or personal informations as the name or the what it can to do) (e.g., asking about programming, sports, movies, or general knowledge). False if the query is related to the medical domain or searching on a human reports or informations."
    )

class Agent():
    """description"""

    def __init__(self, API, build_agent=True):

        self.patient_data = pd.read_csv('data/patient_data2.csv')
        with open('data/prompt2.txt', 'rb') as f:
            self.prompt = f.read()
        self.message_history = {
                    "messages": [
                        ("system", self.prompt),
                    ]
                }
        embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        vector_database = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embedding_model
        )

        self.retriever = vector_database.as_retriever(search_type="similarity", search_kwargs={'k': 4})
        self.recognizer = sr.Recognizer()

        @ tool
        def search_patient(name:str, age:int, gender:str, Patient_ID:int, bed:int) -> dict:
            """search on the patients data, NOTE you must all the valus if there is a missing data you can add "''" for the string variables and "0" for the numerical variables"""
            df2 = self.patient_data.copy()
            if name:
                df2 = df2[df2['Name'].str.contains(name, case=False)]
            if age:
                df2 = df2[df2['Age']==age]
            if Patient_ID:
                df2 = df2[df2['PatientID']==Patient_ID]
            if gender:
                df2 = df2[df2['PatientSex']==gender]
            if bed:
                df2 = df2[df2['Bed']==bed]
            return df2.to_json(orient="records")
        
        @ tool
        def search_query(query:str) -> list:
            """
            search in the medical database for medical informations
            # it return the most similar 4 documents from the Book "Initial management for acute coronary syndrome"
            """
            retrieved_docs = self.retriever.invoke(query)
            return retrieved_docs

        @tool
        def show_patient_report(ImageID:str, StudyDate:int, StudyID:int, PatientID:int, PatientBirth:int, PatientSex:str, ViewPosition_DICOM:str, ReportID:int,
       Report:str, MethodLabel:str, Labels:list[str], Name:str, Age:int, Bed:int) -> str:
            """Adding the patient report dictionary, when you use these tool the report will be showed in the web app"""
            patient_report = {
                "ImageID":ImageID,
                "StudyDate":StudyDate,
                "StudyID":StudyID,
                "PatientID":PatientID,
                "PatientBirth":PatientBirth,
                "PatientSex":PatientSex,
                "ViewPosition_DICOM":ViewPosition_DICOM,
                "ReportID":ReportID,
                "Report":Report,
                "MethodLabel":MethodLabel,
                "Labels":Labels,
                "Name":Name,
                "Age":Age,
                "Bed":Bed,
            }
            return " Added the patient report successfully"

        def create_agent(API):
            """create the agent and return it"""
            tools = [search_patient, search_query, show_patient_report]
            # "gemini-3.1-flash-lite-preview"
            llm = ChatGoogleGenerativeAI(
                model="gemini-3.5-flash-lite",
                google_api_key=API,
                temperature=0
            )
            return create_react_agent(llm, tools)

        if build_agent:
            self.agent = create_agent(API)
        else:
            return create_agent(API), self.message_history

        self.guard_llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite", 
            google_api_key=API,
            temperature=0
        ).with_structured_output(MedicalGuardrail)

    def add_message(self, sender, message):
        """stor message from human or ai into the message history"""
        if sender=='ai':
            self.message_history['messages'].append(('ai',message))
        elif sender=='human':
            self.message_history['messages'].append(('human',message))
        else:
            return f"Value Error '{sender}': Sender must be either 'ai' or 'human' not {sender}!"


    def STT(self, path, language='Arabic'):
        """Speech to text method"""
        # recognizer = sr.Recognizer()
        with sr.AudioFile(path) as source:
            audio = self.recognizer.record(source)
        if language == 'Arabic':
            text = self.recognizer.recognize_google(audio, language="ar-EG")
        else:
            text = self.recognizer.recognize_google(audio, language="en-US")
        return text

    def predict(self, message, voice=False, language='Arabic'):
        """predict agentic response"""
        if voice:
            message = self.STT(message, language)
                # if guard_result.is_diagnosis_attempt:
        self.add_message('human', message)
        guard_prompt = f"Classify this user query: '{message}'"
        guard_result = self.guard_llm.invoke(guard_prompt)
        
        if guard_result.is_out_of_context:
            self.add_message('ai', "عذراً، أنا مساعد ذكاء اصطناعي متخصص في المجال الطبي فقط. يمكنني مساعدتك في البحث عن بيانات المرضى، عرض التقارير الطبية، أو الاستفسار عن الإرشادات والبروتوكولات العلاجية. كيف يمكنني مساعدتك في إحدى هذه المهام؟")
            return self.message_history['messages'][1:], ""
        elif not guard_result.is_diagnosis_attempt:
            
            response = self.agent.invoke(self.message_history)
            final_message = response['messages'][-1]

            if isinstance(final_message.content, list):
                response_text = final_message.content[0].get('text', '')
            else:
                response_text = final_message.content

            self.add_message('ai', response_text)
            patient_report = ""
            for msg in response['messages']:
                if msg.type == 'ai' and hasattr(msg, 'tool_calls') and msg.tool_calls:
                    # query = ""
                    for call in msg.tool_calls:
                        if call['name'] == 'show_patient_report':
                            patient_report = call['args']
            return self.message_history['messages'][1:], patient_report
        else:
            self.add_message("ai","تنبيه مهم: كأنظمة ذكاء اصطناعي، لا يمكنني تقديم تشخيص طبي لحالتك أو كتابة علاج بناءً على الأغراض والتشخيصات الشخصية. يُرجى التوجه إلى أقرب مركز طبي أو استشارة طبيب متخصص فوراً للحصول على التقييم الطبي الصحيح. يمكنك بدلاً من ذلك استفساري عن الإرشادات والبروتوكولات الطبية العامة")
            return self.message_history['messages'][1:], ""