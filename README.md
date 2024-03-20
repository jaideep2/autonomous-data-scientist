# autonomous-data-scientist
Is it possible to automate pareto-optimal data workflows? Possibly. Here's an attempt!


# AI-Powered Machine Learning Wizard

Welcome to the autonomous Machine Learning Wizard. This is first iteration of the problem, a working proof of concept if you will. The current UI is Streamlit application designed to guide you through the process of defining, assessing, and solving machine learning problems. 
## Features

- **Problem Definition**: Clarify your machine learning problem with our Problem Definition Agent.
- **Data Assessment**: Evaluate your dataset's quality and suitability, and receive suggestions for preprocessing steps.
- **Model Recommendation**: Get expert recommendations on the most suitable machine learning models for your problem.
- **Starter Code Generation**: Get generated Python code for data handling, model definition, and a basic training loop.

## How It Works

1. **Define Your Machine Learning Problem**: Describe the machine learning problem you are aiming to solve.
2. **Upload Your Dataset (Optional)**: Provide a sample of your data in .csv format for a more tailored assessment and recommendation.
3. **Receive Customized Guidance and Code**: Based on your input and data, our AI agents will offer specific advice and generate starter Python code to kickstart your ML project.

## Installation

Clone this repository to your local machine:

```bash
git clone <repository-url>
```

Navigate to the project directory:

```bash
cd <project-directory>
```

Install dependencies (feel free to use your favorite virtual env manager, I'm using conda for now but I prefer Poetry ):

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run main.py
```

## Configuration

To utilize the full capabilities of the AI-Powered Machine Learning Wizard, you'll need to provide an API key for the Langchain Groq models. Set up your `secrets.toml` file under `.streamlit` folder with your Groq API key as follows:

```python
# streamlit.secrets example
GROQ_API_KEY = "your_api_key_here"
```

## Usage

Upon launching the application, follow the on-screen instructions to navigate through the different stages of machine learning project development.

## Technology Stack

- **Streamlit**
- **Pandas**
- **CrewAI**: A framework built on top of Langchain for orchestrating AI agents to perform complex tasks. Prod usage might require custom code, but this project isn't there yet.
- **Langchain Groq**: Groq's superfast inference makes it a great fit for agentic workflows or CoT/ToT tasks!

