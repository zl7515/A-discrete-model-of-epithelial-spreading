import numpy as np
from math import *
from matplotlib.pyplot import *
from scipy.spatial import Delaunay


""" Initialisation functions """

def polygon_sort(corners):
    """ This function sorts the cell nodes in sequence. """
    
    number_nodes = len(corners)
    cx = float(sum(x for x, y in corners)) / number_nodes
    cy = float(sum(y for x, y in corners)) / number_nodes

    corners_with_angles = []

    for x, y in corners:
        an = round((np.arctan2(y - cy, x - cx) + 2.0*np.pi) % (2.0*np.pi), 3)
        corners_with_angles.append((x, y, an))

    corners_with_angles.sort(key = lambda tup: tup[2])
    
    return [[round(x*1000)/1000, round(y*1000)/1000] for x, y, z in corners_with_angles]

def initialize_global_parameters():
    """ This function initializes the positions of the cell centers. """
    
    global centers, node_pos, list_nodes_x, list_nodes_y, polarization
    
    # Define centers
    centers = np.zeros((number_centers, 2))
    x0 = 1
    index = 0
    for i in range(nb_layer_y):
        centers[:,0][index:index+nb_layer_x] = x0 + np.zeros(nb_layer_x)
        x0 += 1.5
        index += nb_layer_x
 
    y0 = sqrt(3)/2
    index = 0
    for i in range(nb_layer_y):
        centers[:,1][index:index+nb_layer_x] = np.linspace(y0, y0+(nb_layer_x-1)*sqrt(3), nb_layer_x)
        y0 = y0 + sqrt(3)/2*(-1)**(i)
        index += nb_layer_x

    # Define dictionary node_pos: key is a cell index, value is a list of node positions
    node_pos = {i:[[centers[i][0] - 1, centers[i][1]],
                 [centers[i][0] - 0.5, centers[i][1] + round(sqrt(3)/2,3)],
                 [centers[i][0] + 0.5, centers[i][1] + round(sqrt(3)/2,3)],
                 [centers[i][0] + 1, centers[i][1]],
                 [centers[i][0] + 0.5, centers[i][1] - round(sqrt(3)/2,3)],
                 [centers[i][0] - 0.5, centers[i][1] - round(sqrt(3)/2,3)]] for i in range(number_centers)}
    # Order nodes in sequence in each cell
    for i in range(len(node_pos)):
        node_pos[i] = polygon_sort(node_pos[i])
        
    # Define polarisation 2D-array
    polarization = [[1,0] for i in range(number_centers)]
    polarization = np.array(polarization)
        
    # Define x-axis and y-axis coordinated of each node
    list_nodes_x = [node_pos[i][j][0] for i in range(len(node_pos)) for j in range(len(node_pos[i]))]
    list_nodes_y = [node_pos[i][j][1] for i in range(len(node_pos)) for j in range(len(node_pos[i]))]


def polygon_in(cell_index, node_index): 
    """ This function finds the corresponding cells the node lies in. """
    
    global node_pos
    
    corner = node_pos[cell_index][node_index]
    
    list_cells_node_lies_in = [
            [cell_number, node_number] 
            for cell_number in range(number_centers) 
            for node_number in range(len(node_pos[cell_number])) 
            if np.linalg.norm(np.array(node_pos[cell_number][node_number]) - np.array(corner)) < 10e-3
    ]
    
    return list_cells_node_lies_in


def find_adj_cells():
    """ This function finds and stores the indexes of adjacent cells. """
    
    global node_pos, adj_dict
    
    adj_dict = {i: [] for i in range(number_centers)} 
    
    for i in range(number_centers):
        num_nodes = len(node_pos[i]) # Number of nodes in cell i
        
        for j in range(num_nodes): # Use nodes of cell i to find adjacent cells
            list_cells_node_lies_in = polygon_in(i, j) 
            for index in range(len(list_cells_node_lies_in)):
                adj_dict[i].append(list_cells_node_lies_in[index][0])
        
        adj_dict[i] = list(set(adj_dict[i])) # Remove repeated cell indexes for adjacency list
        adj_dict[i].remove(i) # Remove cell i itself from adjacency list

def unique_nodes():
    """ This function allocates an index to each node and stores the cells it lies in. """
    
    global node_pos, unique_nodes
    
    unique_nodes = {}
    node_index = 0
    list_explored = []
    # Explore each cell
    for cell_index in node_pos:
        cell = node_pos[cell_index]
        # For each node of the cell check if it has already been explored in previous explored cells.
        for i in range(len(cell)):
            # If yes, ignore. If not, add a new node instance and find cells in which the node lies.
            if [cell_index, i] not in list_explored:
                unique_nodes[node_index] = {'cells': polygon_in(cell_index, i)}
                list_explored += unique_nodes[node_index]['cells']
                node_index += 1
                
   
""" Support functions for mesh evolution """

def polygon_perimeter(corners):
    """ This function computes the perimeter of a cell. """
    
    nb_corners = len(corners)
    corners_sorted = polygon_sort(corners)
    perimeter = np.linalg.norm([corners_sorted[0][0]-corners_sorted[nb_corners-1][0],
                                corners_sorted[0][1]-corners_sorted[nb_corners-1][1]])  
    for j in range(nb_corners-1):
        perimeter += np.linalg.norm([corners_sorted[j][0] - corners_sorted[j+1][0],
                                     corners_sorted[j][1] - corners_sorted[j+1][1]])
      
    return perimeter

def polygon_area(corners):
    """ This function computes the area of a cell. """

    corners = polygon_sort(corners)    
    number_nodes = len(corners)
    area = 0.0

    for i in range(number_nodes):
        j = (i + 1) % number_nodes
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]

    return abs(area) / 2.0

def total_area(node_position):
    """ This function computes the area of the whole network. """
    
    number_nodes = len(node_position)
    
    return np.sum([polygon_area(node_pos[i]) for i in range(number_nodes)])


def total_perimeter(node_position):
    """ This function computes the perimeter of the whole network. """
    
    number_nodes = len(node_position)
    
    return np.sum([polygon_perimeter(node_pos[i]) for i in range(number_nodes)])


def polygon_EE(mu, vu, A0i, Ai, Li):   ### elastic energy of individual polygon
    """ This function computes the elastic energy of a cell. """
    
    return 0.5*(mu*Li**2 + vu*(Ai-A0i)**2)

def find_edge_cells(cell_index):
    """ This function decides if the cell on the boundary or not, if so, return node indexes which determine edges. """
    
    global adj_dict, node_pos
    
    adj_cell_index = adj_dict[cell_index]  # Extract adjacent cell indexes
    num_adj_cell = len(adj_cell_index)   # Number of adjacent cells
    num_nodes = len(node_pos[cell_index])  # Number of nodes for this cell
    
    nodes_index = []  # Store index of nodes that lie on edges
    
    # Use number of adjacent cells to classify the position of this cell
    if num_adj_cell == 5:  # If 5 neighbours
        for i in range(num_nodes):
            if len(polygon_in(cell_index, i)) == 2:
                nodes_index.append(i)
    elif 4 >= num_adj_cell >= 2: # If 2, 3 or 4 neighbours
        for i in range(num_nodes):
            if len(polygon_in(cell_index, i)) <= 2:
                nodes_index.append(i)
    
    return nodes_index # empty list if cell is not at the edge. otherwise return sth like [0,1,2,4,5] if only 
                       # node 3 is not the node of any edge that lies on the boundary

def angle_in_0_2pi(angle):
    """ This function changes angle to [0,2pi]. """
    
    return (angle + 2*pi) % (2*pi)

def update_node_position(node, mobility_condition):
    """ Compute resulting displacement and update position of node in each cell it lies in. """
    
    global unique_nodes, node_pos
    
    displacement = dt*K*mobility_condition*unique_nodes[node]['total_force']
    for info_cell_node in unique_nodes[node]['cells']:
        cell_index = info_cell_node[0]
        node_index_in_cell = info_cell_node[1]
        node_pos[cell_index][node_index_in_cell] += displacement


""" Chemical """

def mapk(a, b, beta_m, tau, t):
    """ This function returns MAPK concentration. 
    Since signal concentration is not considered, identical everywhere. """
    
    m0 = 0.1*a/(beta_m*b)   # Set initial condition as 0.1*m_max
    
    return m0*exp(-beta_m*t/tau)
    

""" Polarization """

def polarization(cell_index):
    """ This function calculates the polarization of each cell. 
    In the main code, call F_act_i first, then call this function. """
    
    global node_pos, polarization
    
    F_act = F_act_i(cell_index)  
    F_wett = F_wett_i(cell_index)
    F_sum = kappa*F_act + F_wett 
    
    polarization[cell_index] = D*F_sum + (polarization[cell_index] - D*F_sum)*exp(-beta_p*dt)  # update polarisation of this cell


""" Forces """


def F_mech_j(cell_index, node_index): 
    """ This function calculates F_mech at each node. """
    
    global node_pos
    
    F_mech = np.array([float(0),float(0)]) # F_mech projections in x and y directions
    list_cells_node_lies_in = polygon_in(cell_index, node_index)  # Cell indexes and the corresponding node index
    
    for [ci, ni] in list_cells_node_lies_in: 
         #  Calculates the contribution of each cell involved in the derivative of E
         
         num_nodes = len(node_pos[ci]) # Number of nodes in this cell
         
         if ni == 0:
             p1 = node_pos[ci][num_nodes-1] # Position of one of adjacent node in the cell
             p2 = node_pos[ci][ni] # Position of the node studied
             p3 = node_pos[ci][ni+1] # Position of one of adjacent node in the cell
             
         elif ni == num_nodes-1:
             p1 = node_pos[ci][ni-1] # Position of one of adjacent node in the cell
             p2 = node_pos[ci][ni] # Position of the node studied
             p3 = node_pos[ci][0]  # Position of one of adjacent node in the cell
             
         else:
             p1 = node_pos[ci][ni-1] # Position of one of adjacent node in the cell
             p2 = node_pos[ci][ni] # Position of the node studied
             p3 = node_pos[ci][ni+1] # Position of one of adjacent node in the cell
         
         L = polygon_perimeter(node_pos[ci])
         denom1 = np.linalg.norm([p1[0]-p2[0], p1[1]-p2[1]]) # 1st_denominator = distance p1 to p2
         denom2 = np.linalg.norm([p2[0]-p3[0], p2[1]-p3[1]]) # 2nd_denominator = distance p3 to p2
         
         dL = (np.array(p2) - np.array(p1))/denom1 + (np.array(p2) - np.array(p3))/denom2
         F_j = mu*L*dL # Contribution of this cell
         F_mech += F_j
          
    return -F_mech
          
def F_wett_i(cell_index):
    """ This function calculates wetting force of each cell. """  
    
    global node_pos
    
    F_wett = np.array([0,0])  # F_wett projections in x and y directions
    egde_nodes_indexes = find_edge_cells(cell_index)
    
    if egde_nodes_indexes == []:  # If the cell is not on the edge, F_wett = 0
        return F_wett
    
    else:
        for i in range(len(egde_nodes_indexes)-1):  # Iterate through these nodes
        
            index_node_1 = egde_nodes_indexes[i]
            index_node_2 = egde_nodes_indexes[i+1]
        
            if index_node_2 - index_node_1 == 1:  # Decide if nodes are adjacent
                delta_position = np.array(node_pos[cell_index][index_node_2]) - np.array(node_pos[cell_index][index_node_1])
                length = np.linalg.norm(delta_position)  # Calculate l_jk in F_wett
                current_angle = np.arctan2(delta_position[1], delta_position[0])  # Angle formed by two nodes
                current_angle = angle_in_0_2pi(current_angle)
                normal_angle = -pi/2 + current_angle  # Remember to apply polygon_sort first so that clockwise 
                F_wett = F_wett + np.array([length*cos(normal_angle), length*sin(normal_angle)])
        
        # Deal with the case when indexes 0 and 5 both exist
        if egde_nodes_indexes[len(egde_nodes_indexes)-1] == len(node_pos[cell_index])-1 and egde_nodes_indexes[0] == 0:  
            delta_position = np.array(node_pos[cell_index][0]) - np.array(node_pos[cell_index][len(node_pos[cell_index])-1])
            length = np.linalg.norm(delta_position)  # Calculate l_jk in F_wett
            current_angle = np.arctan2(delta_position[1], delta_position[0])  # Angle formed by two nodes
            current_angle = angle_in_0_2pi(current_angle)  # Change angle to [0,2pi]
            normal_angle = -pi/2 + current_angle  # Remember to apply polygon_sort first so that clockwise 
            F_wett = F_wett+np.array([length*cos(normal_angle), length*sin(normal_angle)])
    
        return FW*F_wett
        
def F_act_i(cell_index):
    """ This function calculates active force of each cell. In main code, first call this, then call polarization. """
    
    global node_pos, adj_dict, polarization
    
    adj_cell_index = adj_dict[cell_index]
    num_adj_cell = len(adj_cell_index)
    F_act = np.array([0,0])
    for i in range(num_adj_cell):
        F_act = F_act + mapk(a, b, beta_m, tau, t)*polarization[i]
    
    return F_act

def total_force_on_cell():
    """ This function computes resulting forces for each cell. """
    
    global node_pos, forces
    
    for cell_index in node_pos:
        cell = node_pos[cell_index]        
        forces[cell_index] = [F_wett_i(cell_index) + F_mech_j(cell_index, i) + F_act_i(cell_index) for i in range(len(cell))]
            
def total_force_on_node(node):
    """ This function computes total force imposed to node by exploring all forces imposed in each cell it lies in. """
    
    global unique_nodes, forces, node_pos, centers
    
    unique_nodes[node]['total_force'] = [0,0]
    total_distance_to_centers = 0
    for info_cell_node in unique_nodes[node]['cells']:
        cell_index = info_cell_node[0]
        node_index_in_cell = info_cell_node[1]
        distance_to_center = np.linalg.norm(node_pos[cell_index][node_index_in_cell]-centers[cell_index])
        unique_nodes[node]['total_force'] += distance_to_center*forces[cell_index][node_index_in_cell]
        total_distance_to_centers += distance_to_center
    unique_nodes[node]['total_force'] /= total_distance_to_centers

""" Mesh visualisation """

def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges    
        
if __name__ == '__main__':
    
    """ Settings """
    
    # Settings
    nb_iterations = 20
    count = 0

    # Dimension parameters
    nb_layer_x = 10
    nb_layer_y = 10
    number_centers = nb_layer_x*nb_layer_y
    
    # Mechanical parameters
    mu = 1
    F0 = 0
    
    # Polarization parameters
    FW = 1
    kappa = 0.1
    D = 0.4
    beta_p=0
    
    #chemical parameters
    a=500
    b=70
    beta_m=1
    tau=0.1
    
    # Speed scale
    K = 1.0
    
    # Time step 
    dt = 0.01  
    t = 0  # needs to be updated
    
    """ Initialisation """
    
    initialize_global_parameters()
    find_adj_cells()
    unique_nodes()
    
    # Visualise initial mesh
    """plt.scatter(list_nodes_x, list_nodes_y, s=10)
    plt.scatter(centers[:,0], centers[:,1], s=5)
    plt.title('Epithelium')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()"""
    
    # Initialize force dictionary
    forces = {}
    
    """ Simulation """

    while nb_iterations > count:
        # Compute resulting forces for each cell
        total_force_on_cell()
        # Go through each node instance
        for node in unique_nodes:
            # Find total force imposed to node by exploring all forces imposed in each cell it lies in.
            total_force_on_node(node)
            # Check if mobility condition is satisfied.
            mobility_condition = np.heaviside(np.linalg.norm(unique_nodes[node]['total_force']) - F0, 0)
            # Compute resulting displacement and update position of node in each cell it lies in.
            update_node_position(node, mobility_condition)
       
        # Reshape cells as numpy arrays
        if count == 0:
            for cell_index in node_pos:
                node_pos[cell_index] = np.stack(node_pos[cell_index], axis=0) 
            
        # Update x-axis and y-axis coordinates of each node
        list_nodes_x = [node_pos[i][j][0] for i in range(len(node_pos)) for j in range(len(node_pos[i]))]
        list_nodes_y = [node_pos[i][j][1] for i in range(len(node_pos)) for j in range(len(node_pos[i]))]  
        
        # Update centers coordinates
        for cell_index in node_pos:
            centers[cell_index] =  [np.mean(node_pos[cell_index][:,0]), np.mean(node_pos[cell_index][:,1])]
        
        # Visualise updated mesh
        figure()
        axis('equal')
        for cell_index in node_pos:
            points = node_pos[cell_index]
            nb_nodes = len(points)
            edges = set([(i, i+1) for i in range(nb_nodes-1)] + [(nb_nodes-1, 0)])
            plot(points[:, 0], points[:, 1], '.')
            for i, j in edges:
                plot(points[[i, j], 0], points[[i, j], 1])
        
        if count==19:
            
            title('$\mu=1$, Fw=1 after 20 iterations')
            savefig('plot3.png')
        show()
        # Move forward in time
        t += dt
        count += 1
        