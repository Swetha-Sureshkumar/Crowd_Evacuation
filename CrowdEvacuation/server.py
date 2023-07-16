import os 
from os import listdir, path
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter


from model import CrowdEvacuation
from agent import FireExit, Wall, Table,Chair, Fire, Smoke, Rescuer, Sight, Door, DeadPerson


# Creates a visual portrayal of our model in the browser interface
#https://github.com/ncsa/COVID19-mesa/blob/master/visualize_feature_per_testing.py
#https://github.com/ncsa/COVID19-mesa/blob/master/visualize_feature.py
#https://github.com/projectmesa/mesa/blob/9a07a48526f11b78e462a9bab390366c95443814/mesa/datacollection.py
#https://github.com/chadsr/MesaFireEvacuation/blob/master/fire_evacuation/agent.py
#https://github.com/tpike3/multilevel_mesa
def crowd_evacuation_portrayal(agent):
    if agent is None:
        return

    portrayal = {
        "x": agent.get_position()[0],
        "y": agent.get_position()[1],
        "scale": 1,
        "Layer": 1
    }

    if isinstance(agent, Rescuer):
        portrayal["scale"] = 1
        portrayal["Layer"] = 5

        mobility = agent.get_mobility()
        if mobility == Rescuer.Mobility.INCAPACITATED:
            portrayal["Shape"] = "resources/materials/incapacitated_human.png"
            portrayal["Layer"] = 6
        elif mobility == Rescuer.Mobility.PANIC:
            portrayal["Shape"] = "resources/materials/panicked_human.png"
        elif agent.is_carrying():
            portrayal["Shape"] = "resources/materials/carrying_human.png"
        else:
            portrayal["Shape"] = "resources/materials/human.png"
    elif isinstance(agent, Fire):
        portrayal["Shape"] = "resources/materials/fire.png"
        portrayal["Layer"] = 3
    elif isinstance(agent, Smoke):
        portrayal["Shape"] = "resources/materials/smoke.png"
        portrayal["Layer"] = 2
    elif isinstance(agent, FireExit):
        portrayal["Shape"] = "resources/materials/fire_exit.png"
        portrayal["Layer"] = 1
    elif isinstance(agent, Door):
        portrayal["Shape"] = "resources/materials/door.png"
        portrayal["Layer"] = 1
    elif isinstance(agent, Wall):
        portrayal["Shape"] = "resources/materials/wall.png"
        portrayal["Layer"] = 1
    elif isinstance(agent, Table):
        portrayal["Shape"] = "resources/materials/table.png"
        portrayal["Layer"] = 1
    elif isinstance(agent, Chair):
        portrayal["Shape"] = "resources/materials/chair.png"
        portrayal["Layer"] = 1
    elif isinstance(agent, DeadPerson):
        portrayal["Shape"] = "resources/materials/dead.png"
        portrayal["Layer"] = 4
    elif isinstance(agent, Sight):
        portrayal["Shape"] = "resources/materials/eye.png"
        portrayal["scale"] = 0.8
        portrayal["Layer"] = 7

    return portrayal



# Was hoping buildingplan could dictate the size of the grid, but seems the grid needs to be specified first, so the size is fixed to 50x50
canvas_element = CanvasGrid(crowd_evacuation_portrayal, 50, 50, 800, 800)

# Define the charts on our web interface visualisation
status_chart = ChartModule(
    [
        {"Label": "Alive", "Color": "green"},
        {"Label": "Dead", "Color": "black"},
        {"Label": "Escaped", "Color": "blue"},
    ]
)

mobility_chart = ChartModule(
    [
        {"Label": "Normal", "Color": "green"},
        {"Label": "Panic", "Color": "yellow"},
        {"Label": "Incapacitated", "Color": "purple"},
    ]
)

rescuing_chart = ChartModule(
    [
        {"Label": "Verbal Rescuing", "Color": "orange"},
        {"Label": "Physical Rescuing", "Color": "red"},
        {"Label": "Morale Rescuing", "Color": "pink"},
    ]
)

# Retrieve the available building plans.
floor_directory = "resources/floors"
floor_plans = [
    f
    for f in os.listdir(floor_directory)
    if os.path.isfile(os.path.join(floor_directory, f))
]

#Fixed parameters used in the model
model_params = {
    "floor_plan_file": UserSettableParameter(
        "choice",
        "buildingplan",
        value=floor_plans[0],
        choices=floor_plans
    ),
    "human_count": UserSettableParameter(
        "number",
        "Number Of Human",
        value=100
    ),
    "rescuing_percentage": UserSettableParameter(
        "slider",
        "Percentage",
        value=50,
        min_value=0,
        max_value=100,
        step=10
    ),
    "fire_probability": UserSettableParameter(
        "slider",
        "Probability of hazard",
        value=0.1,
        min_value=0,
        max_value=1,
        step=0.01
    ),
    "random_spawn": UserSettableParameter(
        "checkbox",
        "Random Locations",
        value=True
    ),
    "visualise_vision": UserSettableParameter(
        "checkbox",
        "Show Agent",
        value=False
    ),
    "save_plots": UserSettableParameter(
        "checkbox",
        "Save",
        value=True
    ),
}

# Initiate the visual server using the provided model.
server = ModularServer(
    CrowdEvacuation,
    [canvas_element, status_chart, mobility_chart, rescuing_chart],
    "Crowd Evacuation",
    model_params,
)
