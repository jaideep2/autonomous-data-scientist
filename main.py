import streamlit as st
import pandas as pd
import os
from crew import *
from langchain_groq import ChatGroq
from code_editor import code_editor
from codeboxapi import CodeBox


def main():
    # Store the code text in the session state
    if 'code_text' not in st.session_state:
        st.session_state['code_text'] = ''

    st.title('AI-Powered Machine Learning Wizard')
    multiline_text: str = """
    Welcome to Machine Learning Wizard! This intelligent assistant will guide you through the process of defining, assessing, and solving your machine learning problems.
    Our team of AI experts is ready to clarify your problem, evaluate your data, recommend suitable models, and even generate starter Python code to kickstart your project.
    """

    st.markdown(multiline_text, unsafe_allow_html=True)

    # Model selection
    model: str = st.selectbox(
        'Choose a model',
        ['mixtral-8x7b-32768', 'llama2-70b-4096']
    )

    llm: ChatGroq = ChatGroq(
        temperature=0, 
        groq_api_key=st.secrets["GROQ_API_KEY"], 
        model_name=model
    )

    # Form to collect user input
    with st.form("my_form", clear_on_submit=True):
        data_upload = False
        user_question = st.text_input("Describe your ML problem:", value="I want to predict the price of a house based on its features.")
        uploaded_file = st.file_uploader("Upload a sample .csv of your data (optional)")
        if uploaded_file is not None:
            try:
                # Attempt to read the uploaded file as a DataFrame
                df = pd.read_csv(uploaded_file).head(5)
                
                # If successful, set 'task_assess_data' to the task with uploaded data
                data_upload = True
                
                # Display the DataFrame in the app
                st.write("Data successfully uploaded and read as DataFrame:")
                st.dataframe(df)
            except Exception as e:
                st.error(f"Error reading the file: {e}")

        # Initialize agents and tasks
        agents, tasks = define_agents_and_tasks(user_question, llm, df, uploaded_file) if data_upload else define_agents_and_tasks(user_question, llm)
        submit_button = st.form_submit_button("Submit")
    
    st.divider()

    # Run agents and tasks
    if submit_button:
        if agents and tasks:
            crew = Crew(
                agents=agents,
                tasks=tasks,
                verbose=0
            )

            crew_result = crew.kickoff()
            code_text = crew_result.split("```python")[1].split("```")[0] # TODO make it less hacky
            st.session_state['code_text'] = code_text

    # Code editor https://github.com/bouzidanas/streamlit-code-editor
    editor = st.container(border=True)
    custom_btns = [
        {
            "name": "Copy",
            "feather": "Copy",
            "hasText": True,
            "alwaysOn": True,
            "commands": ["copyAll"],
            "style": {"top": "0.46rem", "right": "0.4rem"}
        },
        {
            "name": "Run",
            "feather": "Play",
            "primary": True,
            "hasText": True,
            "alwaysOn": True,
            "showWithIcon": True,
            "commands": ["submit"],
            "style": {"bottom": "0.44rem", "right": "0.4rem"}
        }
    ]
    with editor:
        st.markdown("### Editor")
        response_dict = code_editor(st.session_state['code_text'], height=[2, 500], buttons=custom_btns)

    st.divider()

    # Code Output https://github.com/shroominic/codebox-api
    output = st.container(border=True)
    with output:
        st.markdown("### Output")
        if 'id' in response_dict and len(response_dict['id']) != 0 and response_dict['type'] == "submit":
            st.session_state['code_text'] = response_dict["text"]
        
            with CodeBox() as codebox:
                result = codebox.run(code=st.session_state['code_text'])

            st.code(result, language='bash', line_numbers=False)


if __name__ == "__main__":
    main()

