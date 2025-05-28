import numpy as np
import matplotlib.pyplot as plt
import random
import math
from mpmath import mp
import matplotlib.animation as animation

a = 1.42 ; L = round(np.sqrt(3) * a / 2 ,3) ; size = 14
T = 294; K_b = 1.380649e-23; h = 6.62607015e-34; k_e = 8.99e9; e = 1.602e-19; K2 = k_e * e*e ; p_cst = (2*K_b*T) / h


def graphene_lattice(size): 
    a = 1.42 ; L = round(np.sqrt(3) * a / 2 ,3) 
    xx, yy = 0, 0
    hx, hy = 0, 0
    carbon_atoms = [] ; hollow_sites = [] 
    for y in np.arange(0, size * 2 + 2):
        for x in np.arange(0, (2 * size + 1) / 2):
            # Carbon atoms
            if y % 2 == 0:
                xx += a if x % 2 == 0 else 2 * a
            else:
                xx += np.sqrt(3) * L if x == 0 else (2 * a if x % 2 == 0 else a)
            #carbon_atoms.append((xx , yy))
            carbon_atoms.append((round(xx, 3), round(yy, 3)))

            # Hollow sites
            if x < (2 * size + 1) / 4 - 1 and y != 0 and y != size * 2 + 2 - 1:
                if y % 2 == 0:
                    hx += 2 * a if x == 0 else 3 * a
                else:
                    hx += 7 * a / 2 if x == 0 else 3 * a
                #hollow_sites.append((hx , hy))
                hollow_sites.append((round(hx, 3), round(hy, 3)))

        xx = 0 ; yy += L
        hx = 0 ; hy += L
    return np.array(carbon_atoms), np.array(hollow_sites)

def plot_graphene_lattice_with_Li(carbon_atoms, Li_positions):
    # this fucntion plot carbon_atoms and Li_positions

    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # Plot carbon atoms
    ax.scatter(carbon_atoms[:, 0], carbon_atoms[:, 1], c="black", s=10, label="Carbon Atoms")
    ax.scatter(Li_positions[:, 0], Li_positions[:, 1], c="blue", s=40, label="Lithium Ions")

    # Draw carbon-carbon bonds
    for i, carbon_atom in enumerate(carbon_atoms):
        for j, neighbor_atom in enumerate(carbon_atoms):
            if i != j:
                distance = np.linalg.norm(carbon_atom - neighbor_atom)
                if 1.3 <= distance <= 1.5:
                    ax.plot([carbon_atom[0], neighbor_atom[0]], [carbon_atom[1], neighbor_atom[1]], color="gray", linewidth=1, alpha=0.3)

    ax.set_xlabel("x-axis (A)", fontsize=12)
    ax.set_ylabel("y-axis (A)", fontsize=12)
    ax.set_title("Graphene Lattice", fontsize=14)
    ax.axis("equal")
    ax.legend()
    plt.show()

def PBC(size, site):
    #this function for Periodic Boundary Conditions
    a = 1.42 ; L = round(np.sqrt(3) * a / 2 ,3) ; x,y = site #L = np.sqrt(3) * a / 2 ; x,y = site 
    x_min = a ; x_max = 2*(size-3)*a
    y_min = L ; y_max = 28 * L
    if x < x_min : 
        dx = abs(x_min - x) ; x = x_max - dx
    elif x > x_max :
        dx = abs(x - x_max) ; x = x_min + dx
    if y > y_max :
        dy = abs(y - y_max)
        y = dy
    elif y < y_min :
        dy = abs(y)
        y = y_max - dy
    
    return round(x, 3), round(y, 3)

def find_valid_neighbors(pos, hollow_sites, Li_distribution, size) :
    neighbors = [
        (pos[0] - (3 * a / 2), pos[1] - L),  # h1
        (pos[0] + (3 * a / 2), pos[1] - L),  # h2
        (pos[0] - (3 * a / 2), pos[1] + L),  # h3
        (pos[0] + (3 * a / 2), pos[1] + L),  # h4
        (pos[0], pos[1] - (2 * L)),          # h5
        (pos[0], pos[1] + (2 * L)),          # h6
        ]
    neighbors = [PBC(size, site) for site in neighbors]
    valid_neighbors = []

    idx_neighbors = [None] * len(neighbors)
    valid_idx_neighbors = []
    # search for indexes of neighbors
    for i, neighbor in enumerate(neighbors) :
        for j, site in enumerate(hollow_sites) :
            if np.allclose(neighbor, site, atol=1e-1):
                idx_neighbors[i] = j
                break
    # search for valid neighbors
    for i, idx in enumerate(idx_neighbors) :
        if Li_distribution[idx] is None :
            # add the condition here
            pos2 = hollow_sites[idx]
            if i == 0 : 
                neighbors2 = [ 
                    (pos2[0] - (3 * a / 2), pos2[1] - L),  # h1
                    (pos2[0] + (3 * a / 2), pos2[1] - L),  # h2
                    (pos2[0] - (3 * a / 2), pos2[1] + L),  # h3
                    (pos2[0], pos2[1] - (2 * L)),          # h5
                    (pos2[0], pos2[1] + (2 * L)),          # h6
                ]
                #Add verification
                neighbors2 = [PBC(size, site) for site in neighbors2]
                idx_neighbors2 = [None] * len(neighbors2)
                for k, n2 in enumerate(neighbors2):
                    for j, site in enumerate(hollow_sites):
                        if np.allclose(n2, site, atol=1e-1):
                            idx_neighbors2[k] = j
                            break
                for idx2 in idx_neighbors2:
                    if Li_distribution[idx2] is not None :
                        break
                else :
                    valid_idx_neighbors.append(idx)
                    valid_neighbors.append(hollow_sites[idx])

            elif i == 1 :
                neighbors2 = [
                    (pos2[0] - (3 * a / 2), pos2[1] - L),  # h1
                    (pos2[0] + (3 * a / 2), pos2[1] - L),  # h2
                    (pos2[0] + (3 * a / 2), pos2[1] + L),  # h4
                    (pos2[0], pos2[1] - (2 * L)),          # h5
                    (pos2[0], pos2[1] + (2 * L)),          # h6

                ]
                #Add verification
                neighbors2 = [PBC(size, site) for site in neighbors2]
                idx_neighbors2 = [None] * len(neighbors2)
                for k, n2 in enumerate(neighbors2):
                    for j, site in enumerate(hollow_sites):
                        if np.allclose(n2, site, atol=1e-1):
                            idx_neighbors2[k] = j
                            break
                for idx2 in idx_neighbors2:
                    if Li_distribution[idx2] is not None :
                        break
                else :
                    valid_idx_neighbors.append(idx)
                    valid_neighbors.append(hollow_sites[idx])

            elif i == 2 :
                neighbors2 = [
                    (pos2[0] - (3 * a / 2), pos2[1] - L),  # h1
                    (pos2[0] - (3 * a / 2), pos2[1] + L),  # h3
                    (pos2[0] + (3 * a / 2), pos2[1] + L),  # h4
                    (pos2[0], pos2[1] - (2 * L)),          # h5
                    (pos2[0], pos2[1] + (2 * L)),          # h6
                ]
                #Add verification
                neighbors2 = [PBC(size, site) for site in neighbors2]
                idx_neighbors2 = [None] * len(neighbors2)
                for k, n2 in enumerate(neighbors2):
                    for j, site in enumerate(hollow_sites):
                        if np.allclose(n2, site, atol=1e-1):
                            idx_neighbors2[k] = j
                            break
                for idx2 in idx_neighbors2:
                    if Li_distribution[idx2] is not None :
                        break
                else :
                    valid_idx_neighbors.append(idx)
                    valid_neighbors.append(hollow_sites[idx])

            elif i == 3 :
                neighbors2 = [
                    (pos2[0] + (3 * a / 2), pos2[1] - L),  # h2
                    (pos2[0] - (3 * a / 2), pos2[1] + L),  # h3
                    (pos2[0] + (3 * a / 2), pos2[1] + L),  # h4
                    (pos2[0], pos2[1] - (2 * L)),          # h5
                    (pos2[0], pos2[1] + (2 * L)),          # h6
                ]
                #Add verification
                neighbors2 = [PBC(size, site) for site in neighbors2]
                idx_neighbors2 = [None] * len(neighbors2)
                for k, n2 in enumerate(neighbors2):
                    for j, site in enumerate(hollow_sites):
                        if np.allclose(n2, site, atol=1e-1):
                            idx_neighbors2[k] = j
                            break
                for idx2 in idx_neighbors2:
                    if Li_distribution[idx2] is not None :
                        break
                else :
                    valid_idx_neighbors.append(idx)
                    valid_neighbors.append(hollow_sites[idx])

            elif i == 4 :
                neighbors2 = [
                    (pos2[0] - (3 * a / 2), pos2[1] - L),  # h1
                    (pos2[0] + (3 * a / 2), pos2[1] - L),  # h2
                    (pos2[0] - (3 * a / 2), pos2[1] + L),  # h3
                    (pos2[0] + (3 * a / 2), pos2[1] + L),  # h4
                    (pos2[0], pos2[1] - (2 * L)),          # h5
                ]
                #Add verification
                neighbors2 = [PBC(size, site) for site in neighbors2]
                idx_neighbors2 = [None] * len(neighbors2)
                for k, n2 in enumerate(neighbors2):
                    for j, site in enumerate(hollow_sites):
                        if np.allclose(n2, site, atol=1e-1):
                            idx_neighbors2[k] = j
                            break
                for idx2 in idx_neighbors2:
                    if Li_distribution[idx2] is not None :
                        break
                else :
                    valid_idx_neighbors.append(idx)
                    valid_neighbors.append(hollow_sites[idx])
            else : 
                neighbors2 = [
                    (pos2[0] - (3 * a / 2), pos2[1] - L),  # h1
                    (pos2[0] + (3 * a / 2), pos2[1] - L),  # h2
                    (pos2[0] - (3 * a / 2), pos2[1] + L),  # h3
                    (pos2[0] + (3 * a / 2), pos2[1] + L),  # h4
                    (pos2[0], pos2[1] + (2 * L)),          # h6

                ] 
                #Add verification
                neighbors2 = [PBC(size, site) for site in neighbors2]
                idx_neighbors2 = [None] * len(neighbors2)
                for k, n2 in enumerate(neighbors2):
                    for j, site in enumerate(hollow_sites):
                        if np.allclose(n2, site, atol=1e-1):
                            idx_neighbors2[k] = j
                            break
                for idx2 in idx_neighbors2:
                    if Li_distribution[idx2] is not None :
                        break
                else :
                    valid_idx_neighbors.append(idx)
                    valid_neighbors.append(hollow_sites[idx])
    return valid_neighbors , valid_idx_neighbors

def find_CLMB_interaction(pos, hollow_sites, Li_distribution, size) :
    clmb_intr=[
        (pos[0] - (3 * a / 2), pos[1] - L),  # h1
        (pos[0] + (3 * a / 2), pos[1] - L),  # h2
        (pos[0] - (3 * a / 2), pos[1] + L),  # h3
        (pos[0] + (3 * a / 2), pos[1] + L),  # h4
        (pos[0], pos[1] - (2 * L)),          # h5
        (pos[0], pos[1] + (2 * L)),          # h6

        (pos[0] - (3 * a / 2), pos[1] - (3 * L)),  # h21
        (pos[0] + (3 * a / 2), pos[1] - (3 * L)),  # h22
        (pos[0] - (3 * a), pos[1]),                # h23
        (pos[0] + (3 * a), pos[1]),                # h24
        (pos[0] - (3 * a / 2), pos[1] + (3 * L)),  # h25
        (pos[0] + (3 * a / 2), pos[1] + (3 * L)),  # h26

        (pos[0], pos[1] - 4 * L),
        (pos[0], pos[1] + 4 * L),
        (pos[0] - 3 * a, pos[1] - 2 * L),
        (pos[0] + 3 * a, pos[1] - 2 * L),
        (pos[0] - 3 * a, pos[1] + 2 * L),
        (pos[0] + 3 * a, pos[1] + 2 * L),
    ]

    clmb_intr = [PBC(size, site) for site in clmb_intr]
    valid_clmb_intr=[]

    idx_clmb_intr = [None] * len(clmb_intr)
    valid_idx_clmb_intr = []
    # search for indexes of clmb_int
    for i, clmb in enumerate(clmb_intr):
        for j, site in enumerate(hollow_sites):
            if np.allclose(clmb, site, atol=1e-1):
                idx_clmb_intr[i] = j
                break
    
    for idx in idx_clmb_intr : 
        if Li_distribution[idx] is not None : #then ther is an interaction 
            valid_idx_clmb_intr.append(idx)
            valid_clmb_intr.append(hollow_sites[idx])
    
    return valid_clmb_intr, valid_idx_clmb_intr

def calc_Transition_Rates(i, hollow_sites, Li_distribution, valid_neighbors, valid_clmb_intr) : 
    T = 294; K_b = 1.380649e-23; h = 6.62607015e-34
    k_e = 8.99e9; e = 1.602e-19; K2 = k_e * e*e
    p_cst = (2*K_b*T) / h

    r_diff = 0 ; sum_rates=0 ; Transition_Rates=[]
    for site in valid_neighbors :
        point_c = (np.array(hollow_sites[i]) + np.array(site)) / 2 
        if len(valid_clmb_intr) == 0 :
            r_diff = 0
        else : 
            for Li in valid_clmb_intr :
                raa = np.linalg.norm(np.array(Li_distribution[i]) - np.array(Li)) ; ra = raa *1e-10
                rcc = np.linalg.norm(np.array(point_c) - np.array(Li)) ; rc = rcc *1e-10
                r_diff += (1/rc) - (1/ra)

        E_m = K2 * r_diff + 0.23*1.602e-19 
        exp = - E_m / ( T * K_b )
        pi = p_cst * mp.exp(exp)
        Transition_Rates.append(pi) 
        sum_rates += pi

    return Transition_Rates, sum_rates

def jump(Transition_Rates, valid_neighbors, valid_idx_neighbors) :
    Cum_Transition_Rates = np.cumsum(Transition_Rates)
    random_nbr = random.random()
    event_selection = random_nbr * Cum_Transition_Rates[-1] 
    
    selected_jump = next(i for i, p in enumerate(Cum_Transition_Rates) if event_selection <= p)

    selected_site = valid_neighbors[selected_jump]
    idx_selected_site = valid_idx_neighbors[selected_jump]

    return selected_site , idx_selected_site

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

carbon_atoms, hollow_sites = graphene_lattice(size)
Li_distribution = [None] * len(hollow_sites)
Li_positions = []

total_Li = 62 #total Li-ions : reservoir

left_edge_sites = [7 + n * 14 for n in range(size)]

Li_positions_history = [] ; Li_trajectories = {}



insertion = 0 ; sim_time = 0.0 ; msd_list = [] ; time_list = []   ; plot_msd_list = []
while insertion < total_Li :
    valid_left_edge_sites = []
    for i in left_edge_sites :
        if Li_distribution[i] is None :
            valid_left_edge_sites.append(i)

    print("*" * 120)

    if len(valid_left_edge_sites) > 0 :
        # if yes : inserte and diffuse 
        # else do not insert but only diffuse
        insertion += 1
        idx = random.choice(valid_left_edge_sites)
        print(f"insertion site = {idx}")
        Li_distribution[idx] = hollow_sites[idx] # I insert a Li ion here 
        Li_trajectories[idx] = [hollow_sites[idx]]

    print(f"insertion = {insertion}")

    i_Li_idx = - 1 ; i_displaced = -1
    position_idexes = [j for j, k in enumerate(Li_distribution) if k is not None]
    print(f"position_idexes = {position_idexes}")

    for i, pos in enumerate(Li_distribution) : # Whenever find a Li ion ==>  Then diffuse
        if pos is not None and i != i_displaced : # I found a Li ion not yeat diffused in this diffusion step
            i_Li_idx += 1
            
            valid_neighbors , valid_idx_neighbors = find_valid_neighbors(pos, hollow_sites, Li_distribution, size)
            valid_clmb_intr, valid_idx_clmb_intr = find_CLMB_interaction(pos, hollow_sites, Li_distribution, size)

            print(f"XX ==> a Li ion at idx i : {i}") #; print(f"valid_neighbors : {np.array(valid_neighbors)}") ; 
            print(f"valid_idx_neighbors : {valid_idx_neighbors}") #; print(f"valid_clmb_intr : {np.array(valid_clmb_intr)}") ; 
            print(f"valid_idx_clmb_intr : {valid_idx_clmb_intr}") 

            if len(valid_neighbors) == 0 : 
                print("X"*120) ; print(f"No Availible neighbors ==> This Li ion can't jump") ; print("X"*120)
                continue

            else : 
                Transition_Rates, sum_rates = calc_Transition_Rates(i, hollow_sites, Li_distribution, valid_neighbors, valid_clmb_intr)
            
                selected_site , idx_selected_site = jump(Transition_Rates, valid_neighbors, valid_idx_neighbors)
                ### selected_site is the same as hollow_sites[idx_selected_site] 
                i_displaced = idx_selected_site

                Li_distribution[idx_selected_site] = hollow_sites[idx_selected_site]
                Li_distribution[i] = None

                if i in Li_trajectories:
                    Li_trajectories[i].append(hollow_sites[idx_selected_site])
                else:
                    Li_trajectories[i] = [hollow_sites[idx_selected_site]]
                
                print(f"This Li ion is displaced to site idexed = {idx_selected_site} of coordinates : {selected_site} = {hollow_sites[idx_selected_site]}\n")


    
    #Li_positions = [pos for pos in Li_distribution if pos is not None]
    
    Li_positions = np.array([pos for pos in Li_distribution if pos is not None])
    Li_positions_history.append(Li_positions)

    #if insertion == 2 : break



plot_graphene_lattice_with_Li(carbon_atoms, np.array(Li_positions))