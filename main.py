from client import *
from simulation import *
domain = 100 #100x100x100
graph_state = GraphState(active_z=domain//2)

simulation = Simulation(domain, StandardBoundaryCondition())
# simulation.charges.append(ChargedBox((20,50,5),(80,52,90),100))
# simulation.charges.append(ChargedBox((20,20,5),(80,22,90),-100))
# simulation.conductors.append(Sphere((50,50,30),10,100))
simulation.charges.append(ChargedSphere((40, 50, 30), 3, 100))
simulation.charges.append(ChargedSphere((30, 70, 30), 3, -100))

# simulation.conductors.append(Sphere((60,65,30),10,100))
# simulation.conductors.append(Sphere((40,30,30),20,-10))
# simulation.conductors.append(Sphere((60,60,30),10,0))
# simulation.conductors.append(Sphere((70,60,30),10,100))
# simulation.dieletrics.append(LinearDielectricBox((20,22,5),(50,50,90),5))
# simulation.dieletrics.append(LinearDielectricSphere((30,50,30),30,10))
grid, potential = simulation.solve(100)

step = 5
t = threading.Thread(target=listen, args=(grid, graph_state, potential), daemon=True)
t.start()
while True:
    graph(simulation, domain, grid, step,graph_state, potential)

