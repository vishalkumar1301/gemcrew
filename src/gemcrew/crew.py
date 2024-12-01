from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class Gemcrew():
	"""Gemcrew crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	@agent
	def text_analyzer(self) -> Agent:
		return Agent(
			config=self.agents_config['text_analyzer'],
		)

	@agent
	def credibility_verifier(self) -> Agent:
		return Agent(
			config=self.agents_config['credibility_verifier'],
		)

	@agent
	def objective_selector(self) -> Agent:
		return Agent(
			config=self.agents_config['objective_selector'],
		)

	@task
	def analyze_text_task(self) -> Task:
		return Task(
			config=self.tasks_config['analyze_text_task'],
		)

	@task
	def verify_credibility_task(self) -> Task:
		return Task(
			config=self.tasks_config['verify_credibility_task'],
			context=[self.analyze_text_task()],
		)

	@task
	def select_objective_task(self) -> Task:
		return Task(
			config=self.tasks_config['select_objective_task'],
			context=[self.verify_credibility_task()],
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the Firstproject crew"""
		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
