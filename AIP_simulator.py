import numpy as np
import pandas as pd
import math as mh
import re
from collections import defaultdict
from transform_block import TransformBlock

path = "C:/Users/Lucas/Desktop/Estudos/Bolsa/VVC_AIP_Simulator/"

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
    return 32 - p, 3 - index

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
        n_state_iidx = 0
        n_state_ifact = 0
        state_iidx = ""
        state_ifact = []
        array_states_iidx = ["Null" for x in range(int(32/state_size))] #it has at most 32/state_size states, because at 32 it starts to repeat
        array_states_ifact = ["Null" for x in range(int(32/state_size))] #it has at most 32/state_size states, because at 32 it starts to repeat
        for iidx,ifact in zip(i,j):
            if(base == iidx):
                state_iidx = state_iidx + "0"
            else:
                state_iidx = state_iidx + "1"
            base = iidx
            count = count + 1
            state_ifact.append(ifact)
            if((count + 1)%state_size == 0):
                if(state_iidx in array_states_iidx):
                    pass
                else:
                    array_states_iidx[n_state_iidx] = state_iidx
                    n_state_iidx = n_state_iidx + 1
                state_iidx = ""
                
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
    
def calculate_MCM_blocks(mode, state_iidx, state_ifact, base = 0, replicate = 0):
    downward_index = base #downward will begin from the base
    constant_vectors = {}
    
    #Number of fases equals to the size of the block 
    
    #Initial fase
    for i,j in zip(state_iidx, range(len(state_iidx))):
        if(int(i) == 0): #base is not changing
           pass
        else:
            if(mode >= 34):
                if(mode >= 50):
                    base = base + 1
                else:
                    base = base - 1
            else:
                #TODO modes < 34
                pass
        
        if(base not in constant_vectors):
            constant_vectors[base] = []
        if((base + 1) not in constant_vectors):
            constant_vectors[base + 1] = []
        if((base + 2) not in constant_vectors):
            constant_vectors[base + 2] = []
        if((base + 3) not in constant_vectors):
            constant_vectors[base + 3] = []
        
        constant_vectors[base].append(str(state_ifact[j]) + "[0]")
        constant_vectors[base + 1].append(str(state_ifact[j]) + "[1]")
        constant_vectors[base + 2].append(str(state_ifact[j]) + "[2]")
        constant_vectors[base + 3].append(str(state_ifact[j]) + "[3]")
    
    downward_constants = []
    for constants in constant_vectors.values():
        downward_constants.append(constants.copy())

    if(mode >= 34):
        if(mode >= 50):
            pass #downward will begin from the base
        else: #negative iidx values only exists to modes < 50
            downward_index = base #downward will begin from the lowest of the negative values
    else:
        #TODO modes < 34
        pass

    #Run throught the remaning fases
    for fase in range(2,len(state_iidx) + 1):
        for i in range(len(downward_constants)):
            index = downward_index + i + 1
            if(index not in constant_vectors):
                constant_vectors[index] = []
            
            for j in downward_constants[i]:
                if(j not in constant_vectors[index] or replicate):
                    constant_vectors[index].append(j)

        downward_index = downward_index + 1   

    for i,j in zip(constant_vectors.values(),constant_vectors.keys()):
        print(j, i)

def calculate_adders(state_iidx, state_ifact, base = 0):
    pass

#calculate_states(modes1, angles1, 64, 4)
#calculate_MCM_blocks(56,"0001",[8,16,24,0])
#calculate_MCM_blocks(56,"0001",[8,16,24,0], 1)
#calculate_MCM_blocks(44,"1000",[24,16,8,0])
calculate_MCM_blocks(56,"00010001",[8,16,24,0,8,16,24,0])
                
                
        




