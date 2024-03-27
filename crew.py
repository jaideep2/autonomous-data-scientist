from crewai import Agent, Task, Crew

def define_agents_and_tasks(user_question, llm, df=None, uploaded_file=None, verbose=True):
    agents = []
    tasks = []

    agents.append(
        Agent(
            role='Problem_Definition_Agent',
            goal="""clarify the machine learning problem the user wants to solve, 
                identifying the type of problem (e.g., classification, regression) and any specific requirements.""",
            backstory="""You are an expert in understanding and defining machine learning problems. 
                Your goal is to extract a clear, concise problem statement from the user's input, 
                ensuring the project starts with a solid foundation.""",
            verbose=verbose,
            allow_delegation=False,
            llm=llm,
        )
    )

    tasks.append(
        Task(
            description="""Clarify and define the machine learning problem, 
                including identifying the problem type and specific requirements.
                
                Here is the user's problem:

                {ml_problem}
                """.format(ml_problem=user_question),
            agent=agents[-1],
            expected_output="A clear and concise definition of the machine learning problem."
            )
    )

    agents.append(
        Agent(
            role='Data_Assessment_Agent',
            goal="""evaluate the data provided by the user, assessing its quality, 
                suitability for the problem, and suggesting preprocessing steps if necessary.""",
            backstory="""You specialize in data evaluation and preprocessing. 
                Your task is to guide the user in preparing their dataset for the machine learning model, 
                including suggestions for data cleaning and augmentation.""",
            verbose=verbose,
            allow_delegation=False,
            llm=llm,
        )
    )

    if df and uploaded_file:
        tasks.append(
            Task(
                description="""Evaluate the user's data for quality and suitability, 
                suggesting preprocessing or augmentation steps if needed.
                
                Here is a sample of the user's data:

                {df}

                The file name is called {uploaded_file}
                
                """.format(df=df.head(),uploaded_file=uploaded_file),
                agent=agents[-1],
                expected_output="An assessment of the data's quality and suitability, with suggestions for preprocessing or augmentation if necessary."
            )
        )
    else:
        tasks.append(
            Task(
                description="""The user has not uploaded any specific data for this problem,
                but please go ahead and consider a hypothetical dataset that might be useful
                for their machine learning problem. 
                """,
                agent=agents[-1],
                expected_output="A hypothetical dataset that might be useful for the user's machine learning problem, along with any necessary preprocessing steps."
            )
        )

    agents.append(
        Agent(
            role='Model_Recommendation_Agent',
            goal="""suggest the most suitable machine learning models based on the problem definition 
                and data assessment, providing reasons for each recommendation.""",
            backstory="""As an expert in machine learning algorithms, you recommend models that best fit 
                the user's problem and data. You provide insights into why certain models may be more effective than others,
                considering classification vs regression and supervised vs unsupervised frameworks.""",
            verbose=verbose,
            allow_delegation=False,
            llm=llm,
        )
    )

    tasks.append(
        Task(
            description="""Suggest suitable machine learning models for the defined problem 
                and assessed data, providing rationale for each suggestion.""",
            agent=agents[-1],
            expected_output="A list of suitable machine learning models for the defined problem and assessed data, along with the rationale for each suggestion."
            )
    )

    agents.append(
        Agent(
            role='Starter_Code_Generator_Agent',
            goal="""generate starter Python code for the project, including data loading, 
                model definition, and a basic training loop, based on findings from the problem definitions,
                data assessment and model recommendation""",
            backstory="""You are a code wizard, able to generate starter code templates that users 
                can customize for their projects. Your goal is to give users a head start in their coding efforts.""",
            verbose=verbose,
            allow_delegation=False,
            llm=llm,
        )
    )

    tasks.append(
        Task(
            description="""Generate starter Python code tailored to the user's project using the model recommendation agent's recommendation(s), 
                including snippets for package import, data handling, model definition, and training
                """,
            agent=agents[-1],
            expected_output="Python code snippets for package import, data handling, model definition, and training, tailored to the user's project, plus a brief summary of the problem and model recommendations."
        )
    )

    return agents, tasks