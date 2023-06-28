import numpy as np
import pandas as pd
import math as mh
import re
from collections import defaultdict
from transform_block import TransformBlock

path = "~/VVC-AIP-Simulator"

modes1 = [34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66]
modes2 = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
modes3 = [2,3,10,18,33,34,35,43,46,49,50,54]
modes_positive = [50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66]
modes_negative = [34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]
angles1 = [-32,-29,-26,-23,-20,-18,-16,-14,-12,-10,-8,-6,-4,-3,-2,-1,0,1,2,3,4,6,8,10,12,14,16,18,20,23,26,29,32]
angles2 = [32,29,26,23,20,18,16,14,12,10,8,6,4,3,2,1,0,-1,-2,-3,-4,-6,-8,-10,-12,-14,-16,-18,-20,-23,-26,-29,-32]
angles3 = [32,29,26,23,20,18,16,14,12,10,8,6,4,3,2,1,0,-1,-2,-3,-4,-6,-8,-10,-12,-14,-16,-18,-20,-23,-26,-29,-32]
angles_positive = [0,1,2,3,4,6,8,10,12,14,16,18,20,23,26,29,32]
angles_negative = [-32,-29,-26,-23,-20,-18,-16,-14,-12,-10,-8,-6,-4,-3,-2,-1,0]

fc_coefficients = {
    "0[0]" : 0, 
    "1[0]" : -1,
    "2[0]" : -2,
    "3[0]" : -2,
    "4[0]" : -2,
    "5[0]" : -3,
    "6[0]" : -4,
    "7[0]" : -4,
    "8[0]" : -4,
    "9[0]" : -5,
    "10[0]" : -6,
    "11[0]" : -6,
    "12[0]" : -6,
    "13[0]" : -5,
    "14[0]" : -4,
    "15[0]" : -4,
    "16[0]" : -4,
    "17[0]" : -4,
    "18[0]" : -4,
    "19[0]" : -4,
    "20[0]" : -4,
    "21[0]" : -3,
    "22[0]" : -2,
    "23[0]" : -2,
    "24[0]" : -2,
    "25[0]" : -2,
    "26[0]" : -2,
    "27[0]" : -2,
    "28[0]" : -2,
    "29[0]" : -1,
    "30[0]" : 0,
    "31[0]" : 0,
    "0[1]" : 64, 
    "1[1]" : 63,
    "2[1]" : 62,
    "3[1]" : 60,
    "4[1]" : 58,
    "5[1]" : 57,
    "6[1]" : 56,
    "7[1]" : 55,
    "8[1]" : 54,
    "9[1]" : 53,
    "10[1]" : 52,
    "11[1]" : 49,
    "12[1]" : 46,
    "13[1]" : 44,
    "14[1]" : 42,
    "15[1]" : 39,
    "16[1]" : 36,
    "17[1]" : 33,
    "18[1]" : 30,
    "19[1]" : 29,
    "20[1]" : 28,
    "21[1]" : 24,
    "22[1]" : 20,
    "23[1]" : 18,
    "24[1]" : 16,
    "25[1]" : 15,
    "26[1]" : 14,
    "27[1]" : 12,
    "28[1]" : 10,
    "29[1]" : 7,
    "30[1]" : 4,
    "31[1]" : 2,
    "0[2]" : 0,
    "0[3]" : 0
}

def simmetry_rule(p, index):
    return str(32 - int(p)), str(3 - int(index))

def calculate_iidx_ifact(modes, angles, size):
    values_ifact = []
    values_iidx = []
    columns = []
    for i,j in zip(modes,angles):
        tb = TransformBlock(size, size, i, j, 0, size*2 + 2, size*2 + 2, 0)
        tb.calculate_constants_mode()
        columns.append(i)
        values_iidx.append(tb.array_iidx)
        values_ifact.append(tb.array_ifact)
       
    df = pd.DataFrame(list(zip(*values_iidx)),columns = columns)
    df.to_excel(excel_writer = path + "values_iidx_" + str(size) + ".xlsx")
    df = pd.DataFrame(list(zip(*values_ifact)),columns = columns)
    df.to_excel(excel_writer = path + "values_ifact_" + str(size) + ".xlsx")



def calculate_samples(modes, angles, size, normalize):
    values_ref = []
    values_id = []
    columns = []
    lowest_id_value = mh.inf
    highest_id_value = -mh.inf
    for i,j in zip(modes,angles):
        tb = TransformBlock(size, size, i, j, 0, size*2 + 2, size*2 + 2, 0)
        tb.calculate_pred_values()
        columns.append(i)
        values_ref.append(tb)
        values_id.append(tb.ref_id)
        if(lowest_id_value > tb.ref_id[0]):
            lowest_id_value = tb.ref_id[0]
        if(highest_id_value < tb.ref_id[-1]):
            highest_id_value = tb.ref_id[-1]

    values_ref_array = []
    for i in values_ref:
        if(normalize):
            i.normalize_ref()
            pass
        values_ref_array.append(i.transform_dict_to_array(lowest_id_value, highest_id_value, normalize))
        
    rows = list(range(lowest_id_value,highest_id_value + 1))

    df = pd.DataFrame(list(zip(*values_ref_array)), index= rows, columns = columns)
    if(normalize):
        df.to_excel(excel_writer = path + "ref_" + str(size) + "_normalized" + ".xlsx")
    else:
        df.to_excel(excel_writer = path + "ref_" + str(size) + ".xlsx")


def calculate_equations(modes, angles, size):
    for i,j in zip(modes,angles):
        tb = TransformBlock(size, size, i, j, 0, size*2 + 2, size*2 + 2, 0)
        tb.calculate_equations_mode()


def calculate_states(modes, angles, block_size, state_size):
    values_ifact = []
    values_iidx = []
    columns = []

    for i,j in zip(modes,angles):
        tb = TransformBlock(block_size, block_size, i, j, 0, block_size*2 + 2, block_size*2 + 2, 0)
        tb.calculate_constants_mode()
        columns.append(i)
        values_iidx.append(tb.array_iidx)
        values_ifact.append(tb.array_ifact)


    array_states_mods_iidx = []
    array_states_mods_ifact = []
    for i,j in zip(values_iidx,values_ifact):
        base = 0
        count = -1
        base_counter = 0
        n_state_iidx = 0
        n_state_ifact = 0
        state_iidx = []
        state_ifact = []
        array_states_iidx = ["Null" for x in range(int(32/state_size))] #it has at most 32/state_size states, because at 32 it starts to repeat
        array_states_ifact = ["Null" for x in range(int(32/state_size))] #it has at most 32/state_size states, because at 32 it starts to repeat
        for iidx,ifact in zip(i,j):
            if(base == iidx):
                state_iidx.append(base_counter)
            else:
                base_counter = base_counter + 1
                state_iidx.append(base_counter)
            base = iidx
            count = count + 1
            state_ifact.append(ifact)
            if((count + 1)%state_size == 0):
                if(state_iidx in array_states_iidx):
                    pass
                else:
                    array_states_iidx[n_state_iidx] = state_iidx
                    n_state_iidx = n_state_iidx + 1
                state_iidx = []
                base_counter = 0
                
            if((count + 1)%state_size == 0):
                if(state_ifact in array_states_ifact):
                    pass
                else:
                    array_states_ifact[n_state_ifact] = state_ifact
                    n_state_ifact = n_state_ifact + 1
                state_ifact = []
                
        array_states_mods_iidx.append(array_states_iidx)
        array_states_mods_ifact.append(array_states_ifact)

    df_iidx = pd.DataFrame(list(zip(*array_states_mods_iidx)), columns = columns)
    df_iidx.to_excel(excel_writer = path + "states_iidx_" + str(block_size) + "_" + str(state_size) + ".xlsx")
    df_ifact = pd.DataFrame(list(zip(*array_states_mods_ifact)), columns = columns)
    df_ifact.to_excel(excel_writer = path + "states_ifact_" + str(block_size) + "_" + str(state_size) + ".xlsx")
    return df_iidx
    
def calculate_MCM_blocks(mode, state_iidx, state_ifact, base = 0, height = 1, replicate = 0, print_values = 0):
    constants_vectors = {}
    variation = 0
    
    #Number of fases equals to the size of the block 
    
    #Initial fase
    downward_index = base #downward will begin from the base
    for i,j in zip(state_iidx, range(len(state_iidx))):
        variation = int(i) - variation
        if(variation == 0):
           pass
        else:
            variation = int(i)
            if(mode >= 34):
                if(mode >= 50):
                    downward_index = base + int(i)
                else:
                    downward_index = base - int(i)
            else:
                #TODO modes < 34
                pass
        
        if(base not in constants_vectors):
            constants_vectors[downward_index] = []
        
        constants_vectors[downward_index].append(str(state_ifact[j]) + "[0]")
        
        if((downward_index + 1) not in constants_vectors):
            constants_vectors[downward_index + 1] = []

        constants_vectors[downward_index + 1].append(str(state_ifact[j]) + "[1]")
    
        if((downward_index + 2) not in constants_vectors):
            constants_vectors[downward_index + 2] = []
        
        constants_vectors[downward_index + 2].append(str(state_ifact[j]) + "[2]")
    
        if((downward_index + 3) not in constants_vectors):
            constants_vectors[downward_index + 3] = []
        
        constants_vectors[downward_index + 3].append(str(state_ifact[j]) + "[3]")
        
    if(height == 1):
        pass #dont need to calculate other lines, only the first fase is necessary
    else:
        downward_constants = []
        for constants in constants_vectors.values():
            downward_constants.append(constants.copy())

        if(mode >= 34):
            if(mode >= 50):
                downward_index = base #downward will begin from the base
            else: #negative iidx values only exists on modes < 50
                pass #downward will begin from the lowest of the negative values
        else:
            #TODO modes < 34
            pass

        #Run throught the remaning fases
        for fase in range(2, height + 1):
            for i in range(len(downward_constants)):
                index = downward_index + i + 1
                if(index not in constants_vectors):
                    constants_vectors[index] = []
                
                for j in downward_constants[i]:
                    if(j not in constants_vectors[index] or replicate):
                        constants_vectors[index].append(j)

            downward_index = downward_index + 1   

    if(print_values):
        print("############################################")
        for i,j in zip(constants_vectors.values(),constants_vectors.keys()):
            print(j, i)
    
    return constants_vectors

def map_to_coefficients(constants_vectors, coefficients, replicate = 0, print_values = 0):
    coefficients_vectors = {}

    for i,j in zip(constants_vectors.values(),constants_vectors.keys()):
        coefficients_vectors[j] = []
        for constant in i:

            if(constant not in coefficients):
                p, index = re.findall(r'\d+', constant) #get p[index] from string containing and put it in two separately variables
                p, index = simmetry_rule(p, index) #transform in a value that exists in the coefficients by the simmetry rule
                value = coefficients[p + '[' + index + ']']
            else:
                value = coefficients[constant]
            
            if(value not in coefficients_vectors[j] or replicate):
                    coefficients_vectors[j].append(value)

    if(print_values):
        print("############################################")
        for i,j in zip(coefficients_vectors.values(),coefficients_vectors.keys()):
            print(j, i)

    return coefficients_vectors

#NOT WORKING
def merge_states(constants_or_coefficients_list, replicate = 0, print_values = 0):
    merged_constants_or_coefficients = {}
    for i in constants_or_coefficients_list:
        merged_constants_or_coefficients = merged_constants_or_coefficients | i

    if(print_values):
        print("############################################")
        for i,j in zip(merged_constants_or_coefficients.values(),merged_constants_or_coefficients.keys()):
            print(j, i)

def calculate_adders(state_iidx, state_ifact, base = 0):
    pass

#calculate_states(modes1, angles1, 64, 32)
#calculate_MCM_blocks(56,"0001",[8,16,24,0])
#calculate_MCM_blocks(56,"0001",[8,16,24,0], 1)
#calculate_MCM_blocks(44,"1000",[24,16,8,0])
#calculate_MCM_blocks(56,"00010001",[8,16,24,0,8,16,24,0], height = 8, print_values = 1)
#calculate_MCM_blocks(56,"00010001",[8,16,24,0,8,16,24,0], height = 1, print_values = 1)
#calculate_MCM_blocks(56,"0001",[8,16,24,0], height = 2, print_values = 1)
'''map_to_coefficients(calculate_MCM_blocks(56,"00010001",[8,16,24,0,8,16,24,0]), fc_coefficients, print_values = 1)
map_to_coefficients(calculate_MCM_blocks(35,"11111111",[3, 6, 9, 12, 15, 18, 21, 24]), fc_coefficients, print_values = 1)
map_to_coefficients(calculate_MCM_blocks(35,"11011111",[27, 30, 1, 4, 7, 10, 13, 16]), fc_coefficients, print_values = 1)
map_to_coefficients(calculate_MCM_blocks(35,"11111011",[19, 22, 25, 28, 31, 2, 5, 8]), fc_coefficients, print_values = 1)
map_to_coefficients(calculate_MCM_blocks(35,"11111110",[11, 14, 17, 20, 23, 26, 29, 0]), fc_coefficients, print_values = 1)
coef = []
coef.append(map_to_coefficients(calculate_MCM_blocks(35,"11111111",[3, 6, 9, 12, 15, 18, 21, 24]), fc_coefficients))
coef.append(map_to_coefficients(calculate_MCM_blocks(35,"11011111",[27, 30, 1, 4, 7, 10, 13, 16]), fc_coefficients))
coef.append(map_to_coefficients(calculate_MCM_blocks(35,"11111011",[19, 22, 25, 28, 31, 2, 5, 8]), fc_coefficients))
coef.append(map_to_coefficients(calculate_MCM_blocks(35,"11111110",[11, 14, 17, 20, 23, 26, 29, 0]), fc_coefficients))'''

#merge_states(coef, print_values = 1)

'''calculate_MCM_blocks(59,"00112233",[14, 28, 10, 24, 6, 20, 2, 16], height = 1, print_values = 1)
calculate_MCM_blocks(59,"01122334",[30, 12, 26, 8, 22, 4, 18, 0], height = 1, base = 3, print_values = 1)
calculate_MCM_blocks(59,"00112233",[14, 28, 10, 24, 6, 20, 2, 16], height = 1, base = 7, print_values = 1)
calculate_MCM_blocks(59,"01122334",[30, 12, 26, 8, 22, 4, 18, 0], height = 1, base = 10, print_values = 1)'''


#map_to_coefficients(calculate_MCM_blocks(59,"00112233",[14, 28, 10, 24, 6, 20, 2, 16], height = 1), fc_coefficients, print_values = 1)    
#map_to_coefficients(calculate_MCM_blocks(59,"01122334",[30, 12, 26, 8, 22, 4, 18, 0], height = 1, base = 3,), fc_coefficients, print_values = 1)
#map_to_coefficients(calculate_MCM_blocks(59,"00112233",[14, 28, 10, 24, 6, 20, 2, 16], height = 1, base = 7), fc_coefficients, print_values = 1)    
#map_to_coefficients(calculate_MCM_blocks(59,"01122334",[30, 12, 26, 8, 22, 4, 18, 0], height = 1, base = 10), fc_coefficients, print_values = 1) 
#map_to_coefficients(calculate_MCM_blocks(59,[0,0,1,1,2,2,3,3,3,4,4,5,5,6,6,7,7,7,8,8,9,9,10,10,10,11,11,12,12,13,13,14],[14, 28, 10, 24, 6, 20, 2, 16, 30, 12, 26, 8, 22, 4, 18, 0, 14, 28, 10, 24, 6, 20, 2, 16, 30, 12, 26, 8, 22, 4, 18, 0], height = 1), fc_coefficients, print_values = 1)
#calculate_MCM_blocks(59,[0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 9, 9, 10, 10, 10, 11, 11, 12, 12, 13, 13, 14],[14, 28, 10, 24, 6, 20, 2, 16, 30, 12, 26, 8, 22, 4, 18, 0, 14, 28, 10, 24, 6, 20, 2, 16, 30, 12, 26, 8, 22, 4, 18, 0], height = 1, print_values = 1) 
#calculate_MCM_blocks(59,[0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 9, 9, 10, 10, 10, 11, 11, 12, 12, 13, 13, 14],[14, 28, 10, 24, 6, 20, 2, 16, 30, 12, 26, 8, 22, 4, 18, 0, 14, 28, 10, 24, 6, 20, 2, 16, 30, 12, 26, 8, 22, 4, 18, 0], height = 1, print_values = 1)          
#calculate_MCM_blocks(51,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0], height = 1, print_values = 1)        
#map_to_coefficients(calculate_MCM_blocks(51,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0], height = 1),fc_coefficients ,print_values = 1)     

#calculate_MCM_blocks(51,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], height = 1, print_values = 1)
#calculate_MCM_blocks(51,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0], height = 1, base = 0,print_values = 1)
map_to_coefficients(calculate_MCM_blocks(51,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], height = 1), fc_coefficients ,print_values = 1)
map_to_coefficients(calculate_MCM_blocks(51,[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], height = 1), fc_coefficients ,print_values = 1)


