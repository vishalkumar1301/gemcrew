from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

from pydantic import BaseModel, Field


class Point(BaseModel):
	topic: str = Field(..., description="Name of the topic.")
	exact_claim: str = Field(..., description="Exact claim being made.")
	description: str = Field(..., description="Brief description of the factual claim.")
	context: str = Field(..., description="General context of the statement, so that the verifier can understand the background of the claim.")
	verification_points: list[str] = Field(..., description="Specific question or detail to verify.")

class Points(BaseModel):
	points: list[Point] = Field(..., description="List of points that should be verified for credibility.")

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
			output_json=Points,
		)

	@task
	def verify_credibility_task(self) -> Task:
		return Task(
			config=self.tasks_config['verify_important_task'],
			context=[self.analyze_text_task()],
			output_json=Points,
		)

	@task
	def select_objective_task(self) -> Task:
		return Task(
			config=self.tasks_config['select_objective_task'],
			context=[self.verify_credibility_task()],
			output_json=Points,
			output_file='output.json',
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
