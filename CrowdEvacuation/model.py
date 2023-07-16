import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time 

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import Coordinate, MultiGrid
from mesa.time import RandomActivation

from agent import Rescuer, Wall, FireExit, Table, Fire, Door, Chair

#https://github.com/ncsa/COVID19-mesa/blob/master/visualize_feature_per_testing.py
#https://github.com/ncsa/COVID19-mesa/blob/master/visualize_feature.py
#https://github.com/projectmesa/mesa/blob/9a07a48526f11b78e462a9bab390366c95443814/mesa/datacollection.py
#https://github.com/chadsr/MesaFireEvacuation/blob/master/fire_evacuation/agent.py
#https://github.com/tpike3/multilevel_mesa
class CrowdEvacuation(Model):
    MIN_HEALTH = 0.75
    MAX_HEALTH = 1

    MIN_SPEED = 1
    MAX_SPEED = 2

    MIN_NERVOUSNESS = 1
    MAX_NERVOUSNESS = 10

    MIN_EXPERIENCE = 1
    MAX_EXPERIENCE = 10

    MIN_VISION = 1
    # MAX_VISION is simply the size of the grid

    def __init__(
        self,
        floor_plan_file: str,
        human_count: int,
        rescuing_percentage: float,
        fire_probability: float,
        visualise_vision: bool,
        random_spawn: bool,
        save_plots: bool,
    ):
        super().__init__()

        with open(os.path.join("resources/floors/", floor_plan_file), "rt") as f:
            buildingplan = np.matrix([line.strip().split() for line in f.readlines()])

        # Rotate the buildingplan so it's interpreted as seen in the text file
        buildingplan = np.rot90(buildingplan, 3)

        # Check what dimension our buildingplan is
        width, height = np.shape(buildingplan)

        # Init params
        self.width = width
        self.height = height
        self.human_count = human_count
        self.rescuing_percentage = rescuing_percentage
        self.visualise_vision = visualise_vision
        self.fire_probability = fire_probability
        # Turns to true when a fire has started
        self.fire_started = False
        self.save_plots = save_plots

        # Set up model objects
        self.schedule = RandomActivation(self)

        self.grid = MultiGrid(height, width, torus=False)

        # Used to start a fire at a random Table location
        self.Table: dict[Coordinate, Table] = {}
        self.Chair: dict[Coordinate, Chair] = {}
        self.fire_exits: dict[Coordinate, FireExit] = {}
        self.doors: dict[Coordinate, Door] = {}

        # If random spawn is false, spawn_pos_list will contain the list of possible spawn points according to the buildin gplan
        self.random_spawn = random_spawn
        self.spawn_pos_list: list[Coordinate] = []

        # Load buildingplan objects
        for (x, y), value in np.ndenumerate(buildingplan):
            pos: Coordinate = (x, y)

            value = str(value)
            floor_object = None
            if value == "W":
                floor_object = Wall(pos, self)
            elif value == "E":
                floor_object = FireExit(pos, self)
                self.fire_exits[pos] = floor_object
                self.doors[pos] = floor_object
            elif value == "T":
                floor_object = Table(pos, self)
                self.Table[pos] = floor_object
            elif value == "C":
                floor_object = Chair(pos, self)
                self.Chair[pos] = floor_object    
            elif value == "D":
                floor_object = Door(pos, self)
                self.doors[pos] = floor_object
            elif value == "S":
                self.spawn_pos_list.append(pos)

            if floor_object is not None:
                self.grid.place_agent(floor_object, pos)
                self.schedule.add(floor_object)

        #for traversing
        self.graph = nx.Graph()
        for agents, x, y in self.grid.coord_iter():
            pos = (x, y)
            if len(agents) == 0 or not any(not agent.traversable for agent in agents):
                neighbors_pos = self.grid.get_neighborhood(
                    pos, moore=True, include_center=True, radius=1
                )

                for neighbor_pos in neighbors_pos:
                    # If the neighbour position is empty, or no non-traversable contents, add an edge
                    if self.grid.is_cell_empty(neighbor_pos) or not any(
                        not agent.traversable
                        for agent in self.grid.get_cell_list_contents(neighbor_pos)
                    ):
                        self.graph.add_edge(pos, neighbor_pos)

        # get statistics from our model run
        self.datacollector = DataCollector(
            {
                "Alive": lambda m: self.count_human_status(m, Rescuer.Status.ALIVE),
                "Dead": lambda m: self.count_human_status(m, Rescuer.Status.DEAD),
                "Escaped": lambda m: self.count_human_status(m, Rescuer.Status.ESCAPED),
                "Incapacitated": lambda m: self.count_human_mobility(
                    m, Rescuer.Mobility.INCAPACITATED
                ),
                "Normal": lambda m: self.count_human_mobility(m, Rescuer.Mobility.NORMAL),
                "Panic": lambda m: self.count_human_mobility(m, Rescuer.Mobility.PANIC),
                "Verbal Rescuing": lambda m: self.count_human_rescuing(
                    m, Rescuer.Action.VERBAL_SUPPORT
                ),
                "Physical Rescuing": lambda m: self.count_human_rescuing(
                    m, Rescuer.Action.PHYSICAL_SUPPORT
                ),
                "Morale Rescuing": lambda m: self.count_human_rescuing(
                    m, Rescuer.Action.MORALE_SUPPORT
                ),
            }
        )

        # Calculate how many agents will be rescuer
        number_rescuer = int(round(self.human_count * (self.rescuing_percentage / 100)))

        # Start placing human 
        for i in range(0, self.human_count):
            # Place human randomly
            if self.random_spawn:
                pos = self.grid.find_empty()
            # Place human at specified spawn locations
            else:
                pos = np.random.choice(self.spawn_pos_list)

            if pos:
                # Create a random human
                health = np.random.randint(self.MIN_HEALTH * 100, self.MAX_HEALTH * 100) / 100
                speed = np.random.randint(self.MIN_SPEED, self.MAX_SPEED)

                if number_rescuer > 0:
                    rescues = True
                    number_rescuer -= 1
                else:
                    rescues = False

                vision_distribution = [0.0058, 0.0365, 0.0424, 0.9153]
                vision = int(
                    np.random.choice(
                        np.arange(
                            self.MIN_VISION,
                            self.width + 1,
                            (self.width / len(vision_distribution)),
                        ),
                        p=vision_distribution,
                    )
                )

                nervousness_distribution = [
                    0.025,
                    0.025,
                    0.1,
                    0.1,
                    0.1,
                    0.3,
                    0.2,
                    0.1,
                    0.025,
                    0.025,
                ]
                # Distribution with slight higher weighting for above median nervousness from who
                nervousness = int(
                    np.random.choice(
                        range(self.MIN_NERVOUSNESS, self.MAX_NERVOUSNESS + 1),
                        p=nervousness_distribution,
                    )
                )
                # Random choice starting at 1 and up to and including 10
                experience = np.random.randint(self.MIN_EXPERIENCE, self.MAX_EXPERIENCE)

                belief_distribution = [0.9, 0.1]
                believes_alarm = np.random.choice([True, False], p=belief_distribution)

                human = Rescuer(
                    pos,
                    health=health,
                    speed=speed,
                    vision=vision,
                    rescues=rescues,
                    nervousness=nervousness,
                    experience=experience,
                    believes_alarm=believes_alarm,
                    model=self,
                )

                self.grid.place_agent(human, pos)
                self.schedule.add(human)
            else:
                print("No tile empty for human placement!")

        self.running = True


    # Plots line charts of various statistics from a run
    def save_figures(self):
        OUTPUT_DIR = "resources/"

        results = self.datacollector.get_model_vars_dataframe()

        dpi = 100
        fig, axes = plt.subplots(figsize=(1920 / dpi, 1080 / dpi), dpi=dpi, nrows=1, ncols=3)

        status_results = results[["Alive", "Dead", "Escaped"]]
        status_plot = status_results.plot(ax=axes[0])
        status_plot.set_title("Human Status")
        status_plot.set_xlabel("Simulation Step")
        status_plot.set_ylabel("Count")

        mobility_results = results[["Incapacitated", "Normal", "Panic"]]
        mobility_plot = mobility_results.plot(ax=axes[1])
        mobility_plot.set_title("Human Mobility")
        mobility_plot.set_xlabel("Simulation Step")
        mobility_plot.set_ylabel("Count")

        rescuing_results = results[["Verbal Rescuing", "Physical Rescuing", "Morale Rescuing"]]
        rescuing_plot = rescuing_results.plot(ax=axes[2])
        rescuing_plot.set_title("Rescuing Operation")
        rescuing_plot.set_xlabel("Simulation Step")
        rescuing_plot.set_ylabel("Successful Attempts")
        rescuing_plot.set_ylim(ymin=0)

        timestr = time.strftime("%Y%m%d-%H%M%S")
        plt.suptitle(
            "Percentage Rescuing: "
            + str(self.rescuing_percentage)
            + "%, Number of Human: "
            + str(self.human_count),
            fontsize=16,
        )
        plt.savefig(OUTPUT_DIR + "graph/" + timestr + ".png")
        plt.close(fig)


    def start_fire(self):
        if np.random.random() < self.fire_probability:
            fire_Table = np.random.choice(list(self.Table.values()))
            pos = fire_Table.pos
            
            fire_Chair = np.random.choice(list(self.Chair.values()))
            pos = fire_Chair.pos

            fire = Fire(pos, self)
            self.grid.place_agent(fire, pos)
            self.schedule.add(fire)

            self.fire_started = True
            print(f"Fire started at position {pos}")


    def step(self):
        self.schedule.step()
        if not self.fire_started:
            self.start_fire()
        self.datacollector.collect(self)
        if self.count_human_status(self, Rescuer.Status.ALIVE) == 0:
            self.running = False
            if self.save_plots:
                self.save_figures()

    @staticmethod
    def count_human_rescuing(model, rescuing_type):
        """
        Helper method to count the number of rescuings performed by Rescuer agents in the model
        """

        count = 0
        for agent in model.schedule.agents:
            if isinstance(agent, Rescuer):
                if rescuing_type == Rescuer.Action.VERBAL_SUPPORT:
                    count += agent.get_verbal_rescuing_count()
                elif rescuing_type == Rescuer.Action.MORALE_SUPPORT:
                    count += agent.get_morale_rescuing_count()
                elif rescuing_type == Rescuer.Action.PHYSICAL_SUPPORT:
                    count += agent.get_physical_rescuing_count()

        return count

    @staticmethod
    def count_human_status(model, status):
        count_status = sum(1 for agent in model.schedule.agents if isinstance(agent, Rescuer) and agent.get_status() == status)
        return count_status


    @staticmethod
    def count_human_mobility(model, mobility):
        count_mobility = sum(1 for agent in model.schedule.agents if isinstance(agent, Rescuer) and agent.get_mobility() == mobility)
        return count_mobility
