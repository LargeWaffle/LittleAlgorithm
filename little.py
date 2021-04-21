from copy import deepcopy
import math
from timeit import default_timer as timer


def print_matrix(dmatrix):
    """
    Print a matrix
    """

    global NBR_TOWNS

    for i in range(NBR_TOWNS):
        print(i, ':', end=' ')

        for j in range(NBR_TOWNS):
            print(f'{dmatrix[i][j]:.2f}', end=' ')

        print('')


def print_solution(sol, evaluation):
    """
    Print a solution
    """

    print(f'{evaluation:0.2f} : {sol}')


def evaluation_solution(sol):
    """
    Evaluation of a solution
    """

    global DIST

    evaluation = 0

    for i in range(NBR_TOWNS - 1):
        evaluation += DIST[sol[i]][sol[i + 1]]

    evaluation += DIST[sol[NBR_TOWNS - 1]][sol[0]]

    return evaluation


def build_nearest_neighbour():
    """
    Nearest neighbour solution
    """

    global DIST
    global BEST_SOLUTION
    global BEST_EVAL

    # Solution of the nearest neighbour
    next_town = 0
    sol = [0 for _ in range(NBR_TOWNS)]

    # Evaluation of the solution
    sol[0] = 0

    for i in range(NBR_TOWNS - 1):

        town_index = next_town
        minimum = math.inf

        for j in range(NBR_TOWNS):

            if 0 <= DIST[town_index][j] < minimum:

                if j not in sol:
                    minimum = DIST[town_index][j]
                    next_town = j

        sol[i + 1] = next_town

    evaluation = evaluation_solution(sol)

    print('Nearest neighbour:')
    print_solution(sol, evaluation)

    BEST_SOLUTION = sol[:]

    BEST_EVAL = evaluation


def build_solution():
    """
    Build final solution
    """

    global BEST_SOLUTION
    global BEST_EVAL

    solution = []
    indice_cour = 0
    ville_cour = 0

    while indice_cour < NBR_TOWNS:

        solution.append(ville_cour)

        # Test si le cycle est hamiltonien
        for i in range(indice_cour):

            if solution[i] == ville_cour:
                # print('Cycle non hamiltonien')
                return

        # Recherche de la ville suivante
        trouve = 0
        i = 0

        while not trouve and i < NBR_TOWNS:

            if STARTING_TOWN[i] == ville_cour:
                trouve = 1
                ville_cour = ENDING_TOWN[i]

            i += 1

        indice_cour += 1

    evaluation = evaluation_solution(solution)

    if BEST_EVAL > 0 or evaluation < BEST_EVAL:

        BEST_EVAL = evaluation

        for i in range(NBR_TOWNS):
            BEST_SOLUTION[i] = solution[i]

        print('New best solution: ', end='')
        print_solution(solution, BEST_EVAL)


# Little algorithm

def substract_row_mins(dmatrix, matrix_mins):
    # Minimums
    for i in range(NBR_TOWNS):

        local_min = math.inf

        for j in range(NBR_TOWNS):

            if 0 <= dmatrix[i][j] <= local_min:
                local_min = dmatrix[i][j]

        if local_min == math.inf:
            local_min = 0

        matrix_mins.extend([local_min])

    # Subtracts the min
    for i in range(NBR_TOWNS):
        for j in range(NBR_TOWNS):

            if dmatrix[i][j] != -1:
                dmatrix[i][j] -= matrix_mins[i]

    return dmatrix


def substract_col_mins(dmatrix, matrix_mins):
    min_list = []

    # Minimums
    for i in range(NBR_TOWNS):

        local_min = math.inf

        for j in range(NBR_TOWNS):

            if 0 <= dmatrix[j][i] < local_min:
                local_min = dmatrix[j][i]

        if local_min == math.inf:
            local_min = 0

        min_list.extend([local_min])

    # Subtracts the min
    for i in range(NBR_TOWNS):
        for j in range(NBR_TOWNS):

            if dmatrix[j][i] != -1:
                dmatrix[j][i] -= min_list[i]

        if min_list[i] > 0:
            matrix_mins.extend([min_list[i]])

    return dmatrix


def get_penalty(dmatrix, irow, icolumn):
    """
    :return: renvoie la penalite d'un endroit precis
    """
    row_value = col_value = math.inf

    for index in range(NBR_TOWNS):

        if index != irow:
            if 0 <= dmatrix[index][icolumn] < col_value:
                col_value = dmatrix[index][icolumn]

            if col_value == math.inf:
                col_value = 0

        if index != icolumn:
            if 0 <= dmatrix[irow][index] < row_value:
                row_value = dmatrix[irow][index]

            if row_value == math.inf:
                row_value = 0

    return row_value + col_value


def locate_zero(dmatrix):
    """
    :return: indices du zero avec la plus grande penalite
    """
    penalty = -math.inf
    izero, jzero = -1, -1

    for i in range(NBR_TOWNS):
        for j in range(NBR_TOWNS):

            if dmatrix[i][j] == 0:
                val = get_penalty(dmatrix, i, j)

                if val > penalty:
                    penalty = val
                    izero, jzero = i, j

    return izero, jzero


def little_algorithm(origin_matrix, iteration, eval_node_parent):
    """
    Algorithme de Little avec version Little+
    """
    global BEST_EVAL
    global STARTING_TOWN
    global ENDING_TOWN

    if iteration == NBR_TOWNS:
        build_solution()
        return

    # Do the modification on a copy of the distance matrix
    copyd = deepcopy(origin_matrix)

    eval_node_child = eval_node_parent

    matrix_mins = []

    copyd = substract_row_mins(copyd, matrix_mins)
    copyd = substract_col_mins(copyd, matrix_mins)

    # Total of the subtracted values with um of the minimims added
    eval_node_child += sum(matrix_mins)

    # Cut : stop the exploration of this node
    if 0 <= BEST_EVAL <= eval_node_child:
        return

    # Row and column of the zero with the max penalty
    izero, jzero = locate_zero(copyd)

    if (izero, jzero) == (-1, -1):
        # print('Solution infeasible\n')
        return

    # Little+ : on cut les sous-tours
    if jzero == 0 and izero == ENDING_TOWN[iteration - 1]:
        return

    STARTING_TOWN[iteration] = izero
    ENDING_TOWN[iteration] = jzero

    # Do the modification on a copy of the distance matrix
    second_copyd = deepcopy(copyd)

    for index in range(NBR_TOWNS):
        second_copyd[index][jzero] = -1
        second_copyd[izero][index] = -1

    # Stops backtracking
    second_copyd[jzero][izero] = -1

    # Explore left child node according to given choice
    little_algorithm(second_copyd, iteration + 1, eval_node_child)

    # Apply penalty because we do the non-choice
    eval_node_child += get_penalty(second_copyd, izero, jzero)

    # Do the modification on a copy of the distance matrix
    second_copyd = deepcopy(copyd)

    second_copyd[izero][jzero] = -1

    # Explore right child node according to non-choice
    little_algorithm(second_copyd, iteration, eval_node_child)


def main():
    """
     * Starting point of the program
       vvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    """

    global NBR_TOWNS
    global DIST
    global STARTING_TOWN
    global ENDING_TOWN
    global BEST_SOLUTION
    global BEST_EVAL
    global COORD

    # Get number of towns from user
    print('Hello adventurer ! How many towns do you want to visit ? \nInput : ', end='')

    # Global variables of the project

    NBR_TOWNS = int(input())
    DIST = [[0 for _ in range(NBR_TOWNS)] for _ in range(NBR_TOWNS)]  # Distance matrix

    STARTING_TOWN = [None for _ in range(NBR_TOWNS)]  # Each edge has a starting
    ENDING_TOWN = [None for _ in range(NBR_TOWNS)]  # and ending node

    BEST_SOLUTION = [0 for _ in range(NBR_TOWNS)]
    BEST_EVAL = -1.0

    COORD = create_coords(NBR_TOWNS)  # Create coordinates from 'berlin52.tsp' file

    print('\nIT45 - LITTLE ALGORITHM')
    print(f'Checking for {NBR_TOWNS} towns\n')

    BEST_EVAL = -1.0

    # Print problem informations
    print('Points coodinates:')

    for i in range(NBR_TOWNS):
        print(f'Town {i}: X = {COORD[i][0]}, Y = {COORD[i][1]}')
    print('')

    # Calcul de la matrice des distances
    for i in range(NBR_TOWNS):
        for j in range(NBR_TOWNS):

            if i is j:
                DIST[i][j] = -1
            else:
                DIST[i][j] = math.sqrt(pow(COORD[j][0] - COORD[i][0], 2)
                                       + pow(COORD[j][1] - COORD[i][1], 2))

    print('Distance Matrix:')
    print_matrix(DIST)
    print('')

    # Fonction comparative 'on choisit le plus petit à chq fois'
    build_nearest_neighbour()

    iteration = 0
    lowerbound = 0.0

    start = timer()
    little_algorithm(DIST, iteration, lowerbound)
    end = timer()

    print("\nBest solution:")
    print_solution(BEST_SOLUTION, BEST_EVAL)

    print(f'Run time in seconds: {end - start:0.8f}')

    print('\n')
    input('Hit ENTER to end the program... ')


def create_coords(nbtowns):
    """
    Get infos from tsp file
    """
    file_offset = 6
    coordinates = []

    with open('./data/berlin52.tsp', 'r') as data_sheet:
        lines = [file_line.split(' ') for file_line in data_sheet]

    input_lines = lines[file_offset:nbtowns + file_offset]

    for elem in input_lines:
        temp = elem[2].replace('\n', '')
        town_coord = [float(elem[1]), float(temp)]
        coordinates.append(town_coord)

    return coordinates


if __name__ == '__main__':
    main()
