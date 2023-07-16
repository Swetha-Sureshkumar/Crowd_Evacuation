from mesa import Agent
import networkx as nx
import numpy as np
from mesa.space import Coordinate
from copy import deepcopy
import uuid
from enum import IntEnum
from typing import Union
from typing_extensions import Self


#https://github.com/projectmesa/mesa/blob/9a07a48526f11b78e462a9bab390366c95443814/mesa/agent.py
#https://github.com/chadsr/MesaFireEvacuation/blob/master/fire_evacuation/agent.py
#https://github.com/ncsa/COVID19-mesa/blob/master/datacollection.py
#https://github.com/tpike3/CSSS_Mesa_tutorial
#https://github.com/tpike3/multilevel_mesa
class Rescuer(Agent):
    """
    A Rescuer agent, which will attempt to escape from the grid.
    It represents an agent responsible for rescuing individuals and escaping from a grid-like environment
    """

    class Mobility(IntEnum):
        INCAPACITATED = 0
        NORMAL = 1
        PANIC = 2

    class Status(IntEnum):
        DEAD = 0
        ALIVE = 1
        ESCAPED = 2
    
    #possible actions that the agent can take
    class Action(IntEnum):
        PHYSICAL_SUPPORT = 0
        MORALE_SUPPORT = 1
        VERBAL_SUPPORT = 2
        RETREAT = 3
    
    MIN_KNOWLEDGE = 0
    MAX_KNOWLEDGE = 1

    MAX_SHOCK = 1.0
    MIN_SHOCK = 0.0
    
    MIN_HEALTH = 0.0
    MAX_HEALTH = 1.0

    MIN_EXPERIENCE = 1
    MAX_EXPERIENCE = 10

    MIN_SPEED = 0.0
    MAX_SPEED = 2.0
    

    # The value the panic score must reach for an agent to start panic behaviour
    PANIC_THRESHOLD = 0.8

    HEALTH_MODIFIER_FIRE = 0.2
    HEALTH_MODIFIER_SMOKE = 0.005

    SPEED_MODIFIER_FIRE = 2
    SPEED_MODIFIER_SMOKE = 0.1

    # When the health value drops below this value, the agent will being to slow down
    SLOWDOWN_THRESHOLD = 0.5

    # Shock modifiers when encountering certain objects per object, per stepp
    DEFAULT_SHOCK_MODIFIER = -0.1 
    SHOCK_MODIFIER_DEAD_RESCUER = 1.0
    SHOCK_MODIFIER_FIRE = 0.2
    SHOCK_MODIFIER_SMOKE = 0.05
    SHOCK_MODIFIER_AFFECTED_RESCUER = 0.1

    MIN_PUSH_DAMAGE = 0.01
    MAX_PUSH_DAMAGE = 0.2

    def __init__(
        self,
        pos: Coordinate,
        health: float,
        speed: float,
        vision: int,
        rescues: bool,
        nervousness,
        experience,
        believes_alarm: bool,
        model,
    ):
        rand_id = get_random_id()
        super().__init__(rand_id, model)

        self.traversable = False

        self.flammable = True
        self.spreads_smoke = True

        self.pos = pos
        self.visibility = 2
        self.health = health
        self.mobility: Rescuer.Mobility = Rescuer.Mobility.NORMAL
        self.shock: int = self.MIN_SHOCK
        self.speed = speed
        self.vision = vision

        # Boolean specifying whether this agent will attempt rescuing
        self.rescues = rescues

        self.verbal_rescuing_count: int = 0
        self.morale_rescuing_count: int = 0
        self.physical_rescuing_count: int = 0

        self.morale_boost: bool = False
        self.carried: bool = False
        self.carrying: Union(Rescuer, None) = None

        self.knowledge = self.MIN_KNOWLEDGE
        self.nervousness = nervousness
        self.experience = experience
        self.believes_alarm = believes_alarm 
        # Boolean stating whether or not the agent believes the alarm is a real fire 
        self.escaped: bool = False

        # The agent and the observed location (agent, (x, y)) where the agent intends to move.
        self.planned_target: tuple[Agent, Coordinate] = (
            None,
            None,
        )
        
        # An intended action that the agent plans to perform upon reaching their designated target, which can be either "carrying" or "boosting morale".
        self.planned_action: Rescuer.Action = None  
        #The variable self.visible_tiles is a list of tuples, where each tuple consists of a coordinate and a tuple of agents.
        self.visible_tiles: tuple[Coordinate, tuple[Agent]] = []

        # An empty set representing what the agent knows of the building plan
        self.known_tiles: dict[Coordinate, set[Agent]] = {}

        # A set that keeps track of the locations where the agent has already been.
        self.visited_tiles: set[Coordinate] = {self.pos}

    def update_sight_tiles(self, visible_neighborhood):
        if len(self.visible_tiles) > 0:
            # Remove vision tiles
            for pos, _ in self.visible_tiles:
                contents = self.model.grid.get_cell_list_contents(pos)
                for agent in contents:
                    if isinstance(agent, Sight):
                        self.model.grid.remove_agent(agent)

        # Add new vision tiles
        for contents, tile in visible_neighborhood:
            # Don't place if the tile has contents but the agent can't see it
            if self.model.grid.is_cell_empty(tile) or len(contents) > 0:
                sight_object = Sight(tile, self.model)
                self.model.grid.place_agent(sight_object, tile)

    # Implementation of ray-casting, using Bresenham's Line Algorithm, which takes into account smoke and visibility of objects
    def get_visible_tiles(self) -> tuple[Coordinate, tuple[Agent]]:
        neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True, radius=self.vision
        )
        visible_neighborhood = set()

        # A set of already checked tiles, for avoiding repetition and thus increased efficiency
        checked_tiles = set()

        for pos in reversed(neighborhood):
            if pos not in checked_tiles:
                blocked = False
                try:
                    # The number of smoke tiles encountered in the path so far
                    smoke_count = 0  
                    path = Bresenham_line(self.pos, pos)

                    for i, tile in enumerate(path):
                        contents = self.model.grid.get_cell_list_contents(tile)
                        visible_contents = []
                        for obj in contents:
                            if isinstance(obj, Sight):
                                # ignore sight tiles
                                continue
                            elif isinstance(obj, Wall):
                                # We hit a wall, reject rest of path and move to next
                                blocked = True
                                break
                            elif isinstance(obj, Smoke):
                                # We hit a smoke tile, increase the counter
                                smoke_count += 1

                            # If the object has a visibility score greater than the smoke encountered in the path, it's visible
                            if obj.visibility and obj.visibility > smoke_count:
                                visible_contents.append(obj)

                        if blocked:
                            checked_tiles.update(
                                path[i:]
                            )  # Add the rest of the path to checked tiles, since we now know they are not visible
                            break
                        else:
                            # If a wall didn't block the way, add the visible agents at this location
                            checked_tiles.add(
                                tile
                            )  # Add the tile to checked tiles so we don't check it again
                            visible_neighborhood.add((tile, tuple(visible_contents)))

                except Exception as e:
                    print(e)

        if self.model.visualise_vision:
            self.update_sight_tiles(visible_neighborhood)

        return tuple(visible_neighborhood)

    def get_random_target(self, allow_visited=True):
        graph_nodes = self.model.graph.nodes()

        known_pos = set(self.known_tiles.keys())

        # If we are excluding visited tiles, remove the visited_tiles set from the available tiles
        if not allow_visited:
            known_pos -= self.visited_tiles

        traversable_pos = [pos for pos in known_pos if self.location_is_traversable(pos)]

        while not self.planned_target[1]:
            i = np.random.choice(len(traversable_pos))
            target_pos = traversable_pos[i]
            if target_pos in graph_nodes and target_pos != self.pos:
                self.planned_target = (None, target_pos)

    def attempt_exit_plan(self):
        self.planned_target = (None, None)
        fire_exits = set()

        for pos, agents in self.known_tiles.items():
            for agent in agents:
                if isinstance(agent, FireExit):
                    fire_exits.add((agent, pos))

        if len(fire_exits) > 0:
            if len(fire_exits) > 1:  # If there is more than one exit known
                best_distance = None
                for exit, exit_pos in fire_exits:
                    length = len(
                        Bresenham_line(self.pos, exit_pos)
                    )  # Let's use Bresenham's to find the 'closest' exit
                    if not best_distance or length < best_distance:
                        best_distance = length
                        self.planned_target = (exit, exit_pos)

            else:
                self.planned_target = fire_exits.pop()

        else:  # If there's a fire and no fire-escape in sight, try to head for an unvisited door, if no door in sight, move randomly (for now)
            found_door = False
            for pos, contents in self.visible_tiles:
                for agent in contents:
                    if isinstance(agent, Door):
                        found_door = True
                        self.planned_target = (agent, pos)
                        break

                if found_door:
                    break

            # Still didn't find a planned_target, so get a random unvisited target
            if not self.planned_target[1]:
                self.get_random_target(allow_visited=False)

    def get_panic_score(self):
        health_component = 1 / np.exp(self.health / self.nervousness)
        experience_component = 1 / np.exp(self.experience / self.nervousness)

        # Calculate the mean of the components
        panic_score = (health_component + experience_component + self.shock) / 3
        return panic_score

    def incapacitate(self):
        self.stop_carrying()
        self.mobility = Rescuer.Mobility.INCAPACITATED
        self.traversable = True

    def die(self):
        # Store the agent's position of death so we can remove them and place a DeadPerson
        pos = self.pos
        self.model.grid.remove_agent(self)
        dead_self = DeadPerson(pos, self.model)
        self.model.grid.place_agent(dead_self, pos)
        print("Agent died at", pos)

    def health_mobility_rules(self):
        moore_neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=True, radius=1
        )
        contents = self.model.grid.get_cell_list_contents(moore_neighborhood)

        for agent in contents:
            if isinstance(agent, Fire):
                self.health -= self.HEALTH_MODIFIER_FIRE
                self.speed -= self.SPEED_MODIFIER_FIRE
            elif isinstance(agent, Smoke):
                self.health -= self.HEALTH_MODIFIER_SMOKE

                # Start to slow the agent when they drop below 50% health
                if self.health < self.SLOWDOWN_THRESHOLD:
                    self.speed -= self.SPEED_MODIFIER_SMOKE

        # Prevent health and speed from going below 0
        if self.health < self.MIN_HEALTH:
            self.health = self.MIN_HEALTH
        if self.speed < self.MIN_SPEED:
            self.speed = self.MIN_SPEED

        if self.health == self.MIN_HEALTH:
            self.stop_carrying()
            self.die()
        elif self.speed == self.MIN_SPEED:
            self.incapacitate()

    def panic_rules(self):
        if self.morale_boost:
            return

        # Shock will decrease by this amount if no new shock is added
        shock_modifier = self.DEFAULT_SHOCK_MODIFIER
        for _, agents in self.visible_tiles:
            for agent in agents:
                if isinstance(agent, Fire):
                    shock_modifier += self.SHOCK_MODIFIER_FIRE - self.DEFAULT_SHOCK_MODIFIER
                if isinstance(agent, Smoke):
                    shock_modifier += self.SHOCK_MODIFIER_SMOKE - self.DEFAULT_SHOCK_MODIFIER
                if isinstance(agent, DeadPerson):
                    shock_modifier += self.SHOCK_MODIFIER_DEAD_RESCUER - self.DEFAULT_SHOCK_MODIFIER
                if isinstance(agent, Rescuer) and agent.get_mobility() != Rescuer.Mobility.NORMAL:
                    shock_modifier += (
                        self.SHOCK_MODIFIER_AFFECTED_RESCUER - self.DEFAULT_SHOCK_MODIFIER
                    )

    
        if not self.believes_alarm and shock_modifier != self.DEFAULT_SHOCK_MODIFIER:
            self.believes_alarm = True

        self.shock += shock_modifier

        # Keep the shock value between 0 and 1
        if self.shock > self.MAX_SHOCK:
            self.shock = self.MAX_SHOCK
        elif self.shock < self.MIN_SHOCK:
            self.shock = self.MIN_SHOCK

        panic_score = self.get_panic_score()

        if panic_score >= self.PANIC_THRESHOLD:
            self.stop_carrying()
            self.mobility = Rescuer.Mobility.PANIC
            self.known_tiles = {}
            self.knowledge = 0
        elif panic_score < self.PANIC_THRESHOLD and self.mobility == Rescuer.Mobility.PANIC:
            self.mobility = Rescuer.Mobility.NORMAL

    def learn_environment(self):
        if self.knowledge < self.MAX_KNOWLEDGE:  
            new_tiles = 0

            for pos, agents in self.visible_tiles:
                if pos not in self.known_tiles.keys():
                    new_tiles += 1
                self.known_tiles[pos] = set(agents)

            # update the knowledge Attribute accordingly
            total_tiles = self.model.grid.width * self.model.grid.height
            new_knowledge_percentage = new_tiles / total_tiles
            self.knowledge = self.knowledge + new_knowledge_percentage
        

    def get_rescuing_cost(self):
        panic_score = self.get_panic_score()
        total_count = (
            self.verbal_rescuing_count
            + self.morale_rescuing_count
            + self.physical_rescuing_count
        )

        rescuing_component = 1 / np.exp(
            1 / (total_count + 1)
        )
        rescuing_cost = (rescuing_component + panic_score) / 2
        return rescuing_cost

    def test_rescuing(self) -> bool:
        rescuing_cost = self.get_rescuing_cost()

        rand = np.random.random()
        # rescuing if rand is GREATER than our rescuing_cost (Higher rescuing_cost means less likely to be rescued)
        if rand > rescuing_cost:
            return True
        else:
            return False

    def verbal_rescuing(self, target_agent: Self, target_location: Coordinate):
        success = False
        for _, agents in self.visible_tiles:
            for agent in agents:
                if isinstance(agent, Rescuer) and agent.get_mobility() == Rescuer.Mobility.NORMAL:
                    if not agent.believes_alarm:
                        agent.set_believes(True)

                    # Inform the agent of the target location
                    if not target_location in agent.known_tiles:
                        agent.known_tiles[target_location] = set()

                    agent.known_tiles[target_location].add(target_agent)
                    success = True

        if success:
            print("Agent informed others of a fire exit!")
            self.verbal_rescuing_count += 1

    def check_for_rescuing(self):
        # If the agent is carrying someone, they are too occupied to do other rescuing
        if self.carrying:
            return

        if self.test_rescuing():
            for location, visible_agents in self.visible_tiles:
                if self.planned_action:
                    break

                for agent in visible_agents:
                    if isinstance(agent, Rescuer) and not self.planned_action:
                        if agent.get_mobility() == Rescuer.Mobility.INCAPACITATED:
                            # If the agent is incapacitated, help them
                            # Physical rescuing
                            # Plan to move toward the target
                            self.planned_target = (
                                agent,
                                location,
                            )
                            # Plan to carry the agent
                            self.planned_action = Rescuer.Action.PHYSICAL_SUPPORT
                            break
                        elif (
                            agent.get_mobility() == Rescuer.Mobility.PANIC
                            and not self.planned_action
                        ):
                            # Morale rescuing
                            # Plan to move toward the target
                            self.planned_target = (
                                agent,
                                location,
                            )
                            # Plan to do morale rescuing with the agent
                            self.planned_action = Rescuer.Action.MORALE_SUPPORT
                            break
                    elif isinstance(agent, FireExit):
                        # Verbal rescuing
                        self.verbal_rescuing(agent, location)

    def get_next_location(self, path):
        path_length = len(path)
        speed_int = int(np.round(self.speed))

        try:
            if path_length <= speed_int:
                next_location = path[path_length - 1]
            else:
                next_location = path[speed_int]

            next_path = []
            for location in path:
                next_path.append(location)
                if location == next_location:
                    break

            return (next_location, next_path)
        except Exception as e:
            raise Exception(
                f"Failed to get next location: {e}\nPath: {path},\nlen: {path_length},\nSpeed: {self.speed}"
            )

    def get_path(self, graph, target, include_target=True) -> list[Coordinate]:
        path = []
        visible_tiles_pos = [pos for pos, _ in self.visible_tiles]

        try:
            if target in visible_tiles_pos:
                path = nx.shortest_path(graph, self.pos, target)
            else:  
                path = nx.shortest_path(graph, self.pos, target)

                if not include_target:
                    del path[
                        -1
                    ]  

            return list(path)
        except nx.exception.NodeNotFound as e:
            graph_nodes = graph.nodes()

            if target not in graph_nodes:
                contents = self.model.grid.get_cell_list_contents(target)
                print(f"Target node not found! Expected {target}, with contents {contents}")
                return path
            elif self.pos not in graph_nodes:
                contents = self.model.grid.get_cell_list_contents(self.pos)
                raise Exception(
                    f"Current position not found!\nPosition: {self.pos},\nContents: {contents}"
                )
            else:
                raise e

        except nx.exception.NetworkXNoPath as e:
            print(f"No path between nodes! ({self.pos} -> {target})")
            return path

    def location_is_traversable(self, pos) -> bool:
        if not self.model.grid.is_cell_empty(pos):
            contents = self.model.grid.get_cell_list_contents(pos)
            for agent in contents:
                if not agent.traversable:
                    return False

        return True

    def get_retreat_location(self, next_location) -> Coordinate:
        x, y = self.pos
        next_x, next_y = next_location
        diff_x = x - next_x
        diff_y = y - next_y

        retreat_location = (sum([x, diff_x]), sum([y, diff_y]))
        return retreat_location

    def check_retreat(self, next_path, next_location) -> bool:
        # Get the contents of any visible locations in the next path
        visible_path = []
        for visible_pos, _ in self.visible_tiles:
            if visible_pos in next_path:
                visible_path.append(visible_pos)

        visible_contents = self.model.grid.get_cell_list_contents(visible_path)
        for agent in visible_contents:
            if (isinstance(agent, Smoke) and not self.planned_action) or isinstance(agent, Fire):
                # There's a danger in the visible path, so try and retreat in the opposite direction
                # Retreat if there's fire, or smoke (and no rescuing attempt)
                retreat_location = self.get_retreat_location(next_location)

                # Check if retreat location is out of bounds
                if not self.model.grid.out_of_bounds(retreat_location):
                    # Check if the retreat location is also smoke, if so, we are surrounded by smoke, so move randomly
                    contents = self.model.grid.get_cell_list_contents(retreat_location)
                    for agent in contents:
                        if isinstance(agent, Smoke) or isinstance(agent, Fire):
                            self.get_random_target()
                            print("Agent surrounded by smoke and moving randomly")
                            retreat_location = None
                            break

                    if retreat_location:
                        print("Agent retreating opposite to fire/smoke")
                        self.planned_target = (None, retreat_location)
                else:
                    self.get_random_target()

                self.planned_action = Rescuer.Action.RETREAT
                return True

        return False

    def update_target(self):
        # If there was a target agent, check if target has moved or still exists
        planned_agent = self.planned_target[0]
        if planned_agent:
            current_pos = planned_agent.get_position()
            if current_pos and current_pos != self.planned_target[1]:  # Agent has moved
                self.planned_target = (planned_agent, current_pos)
            elif not current_pos:  # Agent no longer exists
                self.planned_target = (None, None)
                self.planned_action = None

    def update_action(self):
        planned_agent, _ = self.planned_target

        if planned_agent:
            # Agent had planned morale rescuing, but the agent is no longer panicking or no longer alive, so drop it.
            if self.planned_action == Rescuer.Action.MORALE_SUPPORT and (
                planned_agent.get_mobility() != Rescuer.Mobility.PANIC
                or not planned_agent.get_status() == Rescuer.Status.ALIVE
            ):
                self.planned_target = (None, None)
                self.planned_action = None
            # Agent had planned physical rescuing, but the agent is no longer incapacitated or has already been carried or is not alive, so drop it.
        elif self.planned_action == Rescuer.Action.PHYSICAL_SUPPORT and (
            (planned_agent.get_mobility() != Rescuer.Mobility.INCAPACITATED)
            or planned_agent.is_carried()
            or planned_agent.get_status() != Rescuer.Status.ALIVE
        ):
            self.planned_target = (None, None)
            self.planned_action = None
        elif self.planned_action == Rescuer.Action.RETREAT:
            return
        else:  # Can no longer perform the action
            self.planned_target = (None, None)
            self.planned_action = None

    def perform_action(self):
        agent, _ = self.planned_target

        if self.planned_action == Rescuer.Action.PHYSICAL_SUPPORT:
            if not agent.is_carried():
                self.carrying = agent
                agent.set_carried(True)
                self.physical_rescuing_count += 1
                print("Agent started carrying another agent")
        elif self.planned_action == Rescuer.Action.MORALE_SUPPORT:
            # Attempt to give the agent a permanent morale boost according to your experience score
            if agent.attempt_morale_boost(self.experience):
                print("Morale boost succeeded")
            else:
                print("Morale boost failed")

            self.morale_rescuing_count += 1

        self.planned_action = None

    def push_Rescuer_agent(self, agent: Self):
        # push the agent to a random 1 square away traversable Coordinate
        neighborhood = self.model.grid.get_neighborhood(
            agent.get_position(),
            moore=True,
            include_center=False,
            radius=1,
        )
        traversable_neighborhood = [
            neighbor_pos
            for neighbor_pos in neighborhood
            if self.location_is_traversable(neighbor_pos)
        ]

        if len(traversable_neighborhood) > 0:
            # push the Rescuer agent to a random traversable position
            i = np.random.choice(len(traversable_neighborhood))
            push_pos = traversable_neighborhood[i]
            print(
                f"Agent {self.unique_id} pushed agent {agent.unique_id} from {agent.pos} to {push_pos}"
            )
            self.model.grid.move_agent(agent, push_pos)

            # inure the pushed agent slightly
            current_health = agent.get_health()
            damage = np.random.uniform(self.MIN_PUSH_DAMAGE, self.MAX_PUSH_DAMAGE)
            agent.set_health(current_health - damage)
        else:
            neighborhood_contents = {}
            for pos in neighborhood:
                neighborhood_contents[pos] = self.model.grid.get_cell_list_contents(pos)
            print(
                f"Could not push agent due to no traversable locations.\nNeighborhood Contents: {neighborhood_contents}"
            )

    def move_toward_target(self):
        next_location: Coordinate = None
        pruned_edges = set()
        graph = deepcopy(self.model.graph)

        self.update_target()
        if self.planned_action: 
            self.update_action()

        while self.planned_target[1] and not next_location:
            if self.location_is_traversable(self.planned_target[1]):
                # Target is traversable
                path = self.get_path(graph, self.planned_target[1])
            else:
                # Target is not traversable (e.g. we are going to another Rescuer), so don't include target in the path
                path = self.get_path(graph, self.planned_target[1], include_target=False)

            if len(path) > 0:
                next_location, next_path = self.get_next_location(path)

                if next_location == self.pos:
                    continue

                if next_location == None:
                    raise Exception("Next location can't be none")

                if self.check_retreat(next_path, next_location):
                    # We are retreating and therefore need to try a totally new path, so continue from the start of the loop
                    continue

                # Test the next location to see if we can move there
                if self.location_is_traversable(next_location):
                    # Move normally
                    self.previous_pos = self.pos
                    self.model.grid.move_agent(self, next_location)
                    self.visited_tiles.add(next_location)

                    if self.carrying:
                        agent = self.carrying
                        if agent.get_status() == Rescuer.Status.DEAD:
                            # Agent is dead, so we can't carry them any more
                            self.stop_carrying()
                        else:
                            # Agent is alive, so try to move them
                            try:
                                self.model.grid.move_agent(self.carrying, self.pos)
                            except Exception as e:
                                agent = self.carrying
                                raise Exception(
                                    f"Failed to move  human:\nException:{e}\nAgent: {agent}\nAgent Position: {agent.get_position()}\nSelf Agent Positon: {self.pos}"
                                )

                elif self.pos == path[-1]:
                    # The Rescuer reached their target!

                    if self.planned_action:
                        self.perform_action()

                    self.planned_target = (None, None)
                    self.planned_action = None
                    break

                else:
                    # We want to move here but it's blocked
                    # check if the location is blocked due to a Rescuer agent
                    pushed = False
                    contents = self.model.grid.get_cell_list_contents(next_location)
                    for agent in contents:
                        # Test the panic value to see if this agent "pushes" the blocking agent aside
                        if (
                            isinstance(agent, Rescuer)
                            and agent.mobility != Rescuer.Mobility.INCAPACITATED
                        ) and (
                            (
                                self.get_panic_score() >= self.PANIC_THRESHOLD
                                and self.mobility == Rescuer.Mobility.NORMAL
                            )
                            or self.mobility == Rescuer.Mobility.PANIC
                        ):
                            # push the agent and then move to the next_location
                            self.push_Rescuer_agent(agent)
                            self.previous_pos = self.pos
                            self.model.grid.move_agent(self, next_location)
                            self.visited_tiles.add(next_location)
                            pushed = True
                            break
                    if pushed:
                        continue

                    # Remove the next location from the temporary graph so we can try pathing again without it
                    edges = graph.edges(next_location)
                    pruned_edges.update(edges)
                    graph.remove_node(next_location)

                    # Reset planned_target if the next location was the end of the path
                    if next_location == path[-1]:
                        next_location = None
                        self.planned_target = (None, None)
                        self.planned_action = None
                        break
                    else:
                        next_location = None

            else:  # No path is possible, so drop the target
                self.planned_target = (None, None)
                self.planned_action = None
                break

        if len(pruned_edges) > 0:
            # Add back the edges we removed when removing any non-traversable nodes from the global graph, because they may be traversable again next step
            graph.add_edges_from(list(pruned_edges))

    def step(self):
        if not self.escaped and self.pos:
            self.health_mobility_rules()

            if self.mobility == Rescuer.Mobility.INCAPACITATED or not self.pos:
                # Incapacitated or died, so return already
                return

            self.visible_tiles = self.get_visible_tiles()

            self.panic_rules()

            self.learn_environment()

            planned_target_agent = self.planned_target[0]

            # If a fire has started and the agent believes it, attempt to plan an exit location if we haven't already and we aren't performing an action
            if self.model.fire_started and self.believes_alarm:
                if not isinstance(planned_target_agent, FireExit) and not self.planned_action:
                    self.attempt_exit_plan()

                # Check if anything in vision can be collaborated with, if the agent has normal mobility
                if self.mobility == Rescuer.Mobility.NORMAL and self.rescues:
                    self.check_for_rescuing()

            planned_pos = self.planned_target[1]
            if not planned_pos:
                self.get_random_target()
            elif self.mobility == Rescuer.Mobility.PANIC:  # Panic
                panic_score = self.get_panic_score()

                if panic_score > 0.9 and np.random.random() < panic_score:
                    # If they have above 90% panic score, test the score to see if they faint
                    print("Agent fainted!")
                    self.incapacitate()
                    return

            self.move_toward_target()

            # Agent reached a fire escape, proceed to exit
            if self.model.fire_started and self.pos in self.model.fire_exits.keys():
                if self.carrying:
                    carried_agent = self.carrying
                    carried_agent.escaped = True
                    self.model.grid.remove_agent(carried_agent)

                self.escaped = True
                self.model.grid.remove_agent(self)

    def get_status(self):
        if self.health > self.MIN_HEALTH and not self.escaped:
            return Rescuer.Status.ALIVE
        elif self.health <= self.MIN_HEALTH and not self.escaped:
            return Rescuer.Status.DEAD
        elif self.escaped:
            return Rescuer.Status.ESCAPED

        return None

    def get_speed(self):
        return self.speed

    def get_mobility(self):
        return self.mobility

    def get_health(self):
        return self.health

    def get_position(self):
        return self.pos

    def get_plan(self):
        return (self.planned_target, self.planned_action)

    def set_plan(self, agent, location):
        self.planned_action = None
        self.planned_target = (agent, location)

    def set_health(self, value: float):
        self.health = value

    def set_believes(self, value: bool):
        if value and not self.believes_alarm:
            print("Agent got the fire alarm!")

        self.believes_alarm = value

    def attempt_morale_boost(self, experience: int) -> bool:
        probability = experience / self.MAX_EXPERIENCE
        rand = np.random.random()
        
        if rand < probability:
            self.morale_boost = True
            self.mobility = Rescuer.Mobility.NORMAL
            return True
        
        return False


    def stop_carrying(self):
        if self.carrying:
            carried_agent = self.carrying
            carried_agent.set_carried(False)
            self.carrying = None
            self.planned_action = None
            print("Agent stopped carrying a person")

    def set_carried(self, value: bool):
        self.carried = value

    def is_carried(self):
        return self.carried

    def is_carrying(self):
        return bool(self.carrying)

    
    def get_verbal_rescuing_count(self):
        return self.verbal_rescuing_count

    def get_morale_rescuing_count(self):
        return self.morale_rescuing_count

    def get_physical_rescuing_count(self):
        return self.physical_rescuing_count



class FloorObjects(Agent):
    """Create a new agent.
       Its contains thing in the floor
    """
    def __init__(
        self,
        pos: Coordinate,
        traversable: bool,
        flammable: bool,
        spreads_smoke: bool,
        visibility: int = 2,
        model=None,
    ):
        rand_id = get_random_id()
        super().__init__(rand_id, model)
        self.pos = pos
        self.traversable = traversable
        self.flammable = flammable
        self.spreads_smoke = spreads_smoke
        self.visibility = visibility

    def get_position(self):
        return self.pos
    

class Sight(FloorObjects):
    def __init__(self, pos, model):
        super().__init__(
            pos, traversable=True, flammable=False, spreads_smoke=True, visibility=-1, model=model
        )

    def get_position(self):
        return self.pos

class Door(FloorObjects):
    def __init__(self, pos, model):
        super().__init__(pos, traversable=True, flammable=False, spreads_smoke=True, model=model)

class Wall(FloorObjects):
    def __init__(self, pos, model):
        super().__init__(pos, traversable=False, flammable=False, spreads_smoke=False, model=model)

class FireExit(FloorObjects):
    def __init__(self, pos, model):
        super().__init__(
            pos, traversable=True, flammable=False, spreads_smoke=False, visibility=6, model=model
        )

class Table(FloorObjects):
    def __init__(self, pos, model):
        super().__init__(pos, traversable=False, flammable=True, spreads_smoke=True, model=model)

class Chair(FloorObjects):
    def __init__(self, pos, model):
        super().__init__(pos, traversable=False, flammable=True, spreads_smoke=True, model=model)

class Fire(FloorObjects):
    """
    A fire agent
    """

    def __init__(self, pos, model):
        super().__init__(
            pos,
            traversable=False,
            flammable=False,
            spreads_smoke=True,
            visibility=20,
            model=model,
        )
        self.smoke_radius = 1

    def step(self):
        neighborhood = self.model.grid.get_neighborhood(
            self.pos, moore=False, include_center=False, radius=self.smoke_radius
        )

        for neighbor_pos in neighborhood:
            contents = self.model.grid.get_cell_list_contents(neighbor_pos)

            if len(contents) > 0:
                has_smoke = False
                has_fire = False
                for agent in contents:
                    if isinstance(agent, Smoke):
                        has_smoke = True
                    elif isinstance(agent, Fire):
                        has_fire = True
                    if has_smoke and has_fire:
                        break

                if not has_fire:
                    for agent in contents:
                        if agent.flammable:
                            fire = Fire(neighbor_pos, self.model)
                            self.model.schedule.add(fire)
                            self.model.grid.place_agent(fire, neighbor_pos)
                            break

                if not has_smoke:
                    for agent in contents:
                        if agent.spreads_smoke:
                            smoke = Smoke(neighbor_pos, self.model)
                            self.model.schedule.add(smoke)
                            self.model.grid.place_agent(smoke, neighbor_pos)
                            break

    def get_position(self):
        return self.pos

class Smoke(FloorObjects):
    """
    A smoke agent
    """
    def __init__(self, pos, model):
        super().__init__(pos, traversable=True, flammable=False, spreads_smoke=False, model=model)
        self.smoke_radius = 1
        # The increment per step to increase self.spread by
        self.spread_rate = 1  
        self.spread_threshold = 1
        # When equal or greater than spread_threshold, the smoke will spread to its neighbors
        self.spread = 0  

    def step(self):
        if self.spread >= 1:
            smoke_neighborhood = self.model.grid.get_neighborhood(
                self.pos, moore=False, include_center=False, radius=self.smoke_radius
            )
            for neighbor in smoke_neighborhood:
                place_smoke = True
                contents = self.model.grid.get_cell_list_contents(neighbor)
                for agent in contents:
                    if not agent.spreads_smoke:
                        place_smoke = False
                        break

                if place_smoke:
                    smoke = Smoke(neighbor, self.model)
                    self.model.grid.place_agent(smoke, neighbor)
                    self.model.schedule.add(smoke)

        if self.spread >= self.spread_threshold:
            self.spread_rate = 0
        else:
            self.spread += self.spread_rate

    def get_position(self):
        return self.pos
    
class DeadPerson(FloorObjects):
    def __init__(self, pos, model):
        super().__init__(pos, traversable=True, flammable=True, spreads_smoke=True, model=model)

def get_random_id() -> uuid.UUID:
    #A unique numeric identified for the agent
    return uuid.uuid4()

def Bresenham_line(start, end):
    """
    Bresenham Line Algorithm uses only integer numbers and integer arithmetic to determine all 
    intermediate points throughout the space between start and end points.
    """
    # Break down start and end tuples 
    x1, y1 = start
    x2, y2 = end

    # Calculate differences
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    # Check if the line is steep
    line_is_steep = dy > dx

    # If the line is steep, rotate it
    if line_is_steep:
        # Swap x and y values for each pair
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # If the start point is further along the x-axis than the end point, swap start and end
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Calculate the differences again
    dx = x2 - x1
    dy = y2 - y1

    # Calculate the error margin
    error_margin = int(dx / 2.0)
    step_y = 1 if y1 < y2 else -1

    # Iterate over the bounding box, generating coordinates between the start and end coordinates
    y = y1
    path = []

    for x in range(x1, x2 + 1):
        # Get a coordinate according to if x and y values were swapped
        coordinate = (y, x) if line_is_steep else (x, y)
        # Add it to our path
        path.append(coordinate)  
        # Deduct the absolute difference of y values from our error_margin
        error_margin -= abs(dy)

        # When the error margin drops below zero, increase y by the step and the error_margin by the x difference
        if error_margin < 0:
            y += step_y
            error_margin += dx

    # The the start and end were swapped, reverse the path
    if swapped:
        path.reverse()

    return path