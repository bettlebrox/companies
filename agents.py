from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool, BrowserbaseLoadTool
import agentops
import newrelic.agent

newrelic.agent.initialize("newrelic.ini")
# YOUR_OTHER_IMPORTS


agentops.init()

search_tool = SerperDevTool()
browser_tool = BrowserbaseLoadTool(text_content=True)
recruiter = Agent(
    role="Senior Recruiter",
    goal="Find startups that have had a recent successful funding\
          round and would be attractive and a good match for your client\
         who wants to work remotely from the GMT timezone",
    backstory=(
        "You're driven by a passion for locating great companies for your clients,"
        "you're passionate about technology at the cutting edge of the AI revolution,"
        "you specialise in finding companies that work remotely including from gmt timezone,"
        "and you're eager to advance you clients careers"
    ),
    memory=True,
    verbose=True,
    allow_delegation=True,
    tools=[search_tool, browser_tool],
    max_rpm=100,
    max_execution_time=90000,
)

researcher = Agent(
    role="Junior Researcher",
    goal="Find companies that match the clients criteria",
    backstory=(
        "This is your first job as a junior researcher, you're eager to learn and grow,"
        "you enjoying using the web to find details about companies that are not obvious,"
        "persistence is your number 1 strength"
    ),
    verbose=True,
    tools=[search_tool, browser_tool],
)

manager = Agent(
    role="Manager",
    goal="Ensure the smooth operation and coordination of the recruitment team",
    verbose=True,
    backstory=(
        "As a seasoned project manager, you excel in organizing "
        "tasks, managing timelines, and ensuring the team stays on track."
        "You have a strong sense of responsibility for delivering high-quality work for the client "
    ),
    allow_code_execution=True,  # Enable code execution for the manager
    allow_delegation=True,  # Allow delegation of tasks to other agents
)


research_task = Task(
    description="Gather a list of startups that might be attractive for you client to join. Your client is interested in startups that have received a series A or series B round of funding the last 6 months, that work remotely, and have people working from the EU already",
    agent=recruiter,
    expected_output="List of startups that fit the clients criteria",
)
analysis_task = Task(
    description="Gather startup details. Specifically location of hq, website, hiring locations, approach to remote work, and other criteria that are relevant for an candidate looking to join. ",
    agent=researcher,
    expected_output="List of details for each startup including reasoning for including in the list",
    depends_on=[research_task],
)

writer = Agent(
    role="Writer",
    goal="Write a report on startups that have might be attractive to candidates",
    backstory="You were previously a recruiter and you enjoy writing reports on companies that make them attractive to candidates",
    verbose=True,
)
writing_task = Task(
    description="Compose a report with a list of interesting startups to for the client to join including their key criteria, location, company growth, industry",
    agent=writer,
    expected_output="A markdown report on startups that have might be attractive to your client",
    output_file="report.md",
    depends_on=[research_task, analysis_task],
)

report_crew = Crew(
    agents=[recruiter, researcher, writer],
    tasks=[research_task, analysis_task, writing_task],
    manager_agent=manager,
    process=Process.hierarchical,
)

result = report_crew.kickoff()

print(result)
