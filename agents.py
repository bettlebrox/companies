from langtrace_python_sdk import langtrace

langtrace.init(
    api_key="b6c03eb5d16caa024a0e0480a0338c11937f1a42758a48024158ffde67e93698"
)
from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool, BrowserbaseLoadTool


search_tool = SerperDevTool()
browser_tool = BrowserbaseLoadTool(text_content=True)
recruiter = Agent(
    role="Senior Recruiter",
    goal="Find startups that would be attractive and a good match for your client's {criteria}",
    backstory=(
        "You're driven by a passion for matching great companies to your clients,"
        "and you're eager to advance your client's career"
    ),
    memory=True,
    verbose=True,
    allow_delegation=True,
    tools=[search_tool, browser_tool],
    max_rpm=100,
    max_execution_time=90000,
)

researcher = Agent(
    role="Senior Researcher",
    goal="Find companies that match the clients criteria",
    backstory=(
        "you enjoying using the web to find details about companies that are not obvious,"
        "persistence is your number 1 strength."
        "For example, you find open roles on a companies website to identify what locations that company normally hires from"
    ),
    verbose=True,
    tools=[search_tool, browser_tool],
)

manager = Agent(
    role="Recruitment Manager",
    goal="Ensure the smooth operation and coordination of the recruitment team",
    verbose=True,
    backstory=(
        "As a seasoned recruitment project manager, you excel in organizing "
        "tasks, managing timelines, and ensuring the team stays on track."
        "You have a strong sense of responsibility for delivering high-quality work for the client "
    ),
    allow_code_execution=True,  # Enable code execution for the manager
    allow_delegation=True,  # Allow delegation of tasks to other agents
)


research_task = Task(
    description="Gather a list of startups that might be attractive for your client to join. Your client has the following criteria: {criteria}",
    agent=recruiter,
    expected_output="List of startups that best fit the clients criteria",
)
analysis_task = Task(
    description="Gather startup details. Specifically location of hq, website, hiring locations, approach to remote work, and the following criteria that are relevant for a candidate looking to join: {criteria} ",
    agent=researcher,
    expected_output="List of details for each startup including reasoning for including in the list",
    depends_on=[research_task],
)

feedback_task = Task(
    description="Discuss the list with the client and prompt for their feedback",
    agent=recruiter,
    human_input=True,
    expected_output="okay to proceed to report or kickoff additional tasks to improve the results",
    allow_delegation=True,
    depends_on=[analysis_task, research_task],
)

writer = Agent(
    role="Writer",
    goal="Write a report on startups that have might be attractive to candidates",
    backstory="You were previously a recruiter and you enjoy writing reports on companies that make them attractive to candidates",
    verbose=True,
)
writing_task = Task(
    description="Compose a report with a list of interesting startups to for the client to join including their key {criteria}",
    agent=writer,
    expected_output="A markdown report on startups that have might be attractive to your client",
    output_file="report.md",
    depends_on=[research_task, analysis_task, feedback_task],
)

report_crew = Crew(
    agents=[recruiter, researcher, writer],
    tasks=[research_task, analysis_task, writing_task, feedback_task],
    manager_agent=manager,
    process=Process.hierarchical,
)

result = report_crew.kickoff(
    inputs={
        "criteria": [
            "b2b SaaS company",
            "most recent round of funding is series B",
            "closed last funding round within the last 2 years",
            "operates remote-first",
            "hires remote in eu",
            "hires product or full-stack engineers with python and react skills",
        ]
    }
)

print(result)
