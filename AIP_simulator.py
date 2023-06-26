import numpy as np
import pandas as pd
import math as mh
import re
from collections import defaultdict

path = "C:/Users/Lucas/Desktop/Estudos/Bolsa/VVC_AIP_Simulator/"

class TransformBlock:
    '''Attributes:
        int nTbW
        int nTbH
        int predModeIntra
        int intraPredAngle
        int refidx
        int refW
        int refH
        int cidx
        int refFilterFlag
        int{ref_index} ref
        int[] ref_id
        int[] array_ifact
        int[] array_iidx
    '''

    
    def __init__(self, nTbW, nTbH, predModeIntra, intraPredAngle, refidx = 0, refW = 0, refH = 0, cidx = 0):
        #Initialize inputs
        self.nTbW = nTbW
        self.nTbH = nTbH
        self.predModeIntra = predModeIntra
        self.intraPredAngle = intraPredAngle
        self.refidx = refidx
        self.refW = refW
        self.refH = refH
        self.cidx = cidx
        

        #refFilterFlag hardcoded for now
        self.refFilterFlag = 1

        #Initialize reference as an empty list
        self.ref = defaultdict(lambda: "Null")
        self.ref_id = []

        #Initialize ifact and iidx array as empty lists
        self.array_ifact = []
        self.array_iidx = []

        self.equations = []
        self.equations_reuse = []

    def calculate_reference_sample_array_greather_equal_34(self):
        index_x = 0
        index_y = - 1 - self.refidx

        #with x = 0...nTbW + refidx + 1. The +1 on the end is to include (nTbW + refidx + 1) in the array
        for x in range((self.nTbW + self.refidx + 1) + 1):
            #ref[x] = p[-1 -refidx + x][-1 -refidx] 
            index_x = -1 - self.refidx + x
            self.ref[x] = "p[" + str(index_x) + "][" + str(index_y) + "]"
            self.ref_id.append(x)

        if (self.intraPredAngle < 0):
            invAngle = round((512*32)/self.intraPredAngle)                
            index_x = - 1 - self.refidx
            index_y = 0

            #with x = -nTbH ... -1
            for x in range(-self.nTbH, 0):
                #ref[x] = p[-1 -refidx][-1 -refidx + Min((x*invAngle + 256) >> 9, nTbH)]
                index_y = -1 - self.refidx + min((x*invAngle + 256) >> 9, self.nTbH)
                self.ref[x] = "p[" + str(index_x) + "][" + str(index_y) + "]"
                self.ref_id.append(x)
                    
        else:
            index_y = - 1 - self.refidx
            #with x = nTbW + 2 + refidx ... refW + refidx
            for x in range(self.nTbW + 2 + self.refidx, (self.refW + self.refidx) + 1):
                #ref[x] = p[-1 -refidx + x][-1 -refidx]
                index_x = -1 - self.refidx + x
                self.ref[x] = ("p[" + str(index_x) + "][" + str(index_y) + "]")
                self.ref_id.append(x)
            
            index_x = -1 + self.refW
            #with x = 1...(Max(1,nTbW/nTbH)*refidx + 1)
            for x in range(1,(max(1,self.nTbW/self.nTbH)*self.refidx + 1) + 1):
                #ref[refW + refidx + x] = p[-1 -refW][-1 -refidx]
                self.ref[self.refW + self.refidx + x] = "p[" + str(index_x) + "][" + str(index_y) + "]"
                self.ref_id.append(x)

        self.ref_id.sort()

    def calculate_reference_sample_array_less_34(self):
        index_x = - 1 - self.refidx
        index_y = 0

        #with x = 0...nTbH + refidx + 1. The +1 on the end is to include (nTbH + refidx + 1) in the array
        for x in range((self.nTbH + self.refidx + 1) + 1):
            #ref[x] = p[-1 -refidx][-1 -refidx + x] 
            index_y = -1 - self.refidx + x
            self.ref[x] = "p[" + str(index_x) + "][" + str(index_y) + "]"
            self.ref_id.append(x)

        if (self.intraPredAngle < 0):
            invAngle = round((512*32)/self.intraPredAngle)                
            index_x = 0
            index_y = - 1 - self.refidx

            #with x = -nTbW ... -1
            for x in range(-self.nTbW, 0):
                #ref[x] = p[-1 -refidx + Min((x*invAngle + 256) >> 9, nTbW][-1 -refidx]
                index_x = -1 - self.refidx + min((x*invAngle + 256) >> 9, self.nTbH)
                self.ref[x] = "p[" + str(index_x) + "][" + str(index_y) + "]"
                self.ref_id.append(x)
                    
        else:
            index_x = - 1 - self.refidx
            #with x = nTbH + 2 + refidx ... refH + refidx
            for x in range(self.nTbH + 2 + self.refidx, (self.refH + self.refidx) + 1):
                #ref[x] = p[-1 -refidx][-1 -refidx + x]
                index_y = -1 - self.refidx + x
                self.ref[x] = ("p[" + str(index_x) + "][" + str(index_y) + "]")
                self.ref_id.append(x)
            
            index_y = -1 + self.refH
            #with x = 1...(Max(1,nTbH/nTbW)*refidx + 1)
            for x in range(1,(max(1,self.nTbH/self.nTbW)*self.refidx + 1) + 1):
                #ref[refH + refidx + x] = p[-1 -refidx][-1 -refH]
                self.ref[self.refH + self.refidx + x] = "p[" + str(index_x) + "][" + str(index_y) + "]"
                self.ref_id.append(x)

        self.ref_id.sort()


    def calculate_pred_values(self):
        if (self.predModeIntra >= 34):
            self.calculate_reference_sample_array_greather_equal_34()
        else:
            self.calculate_reference_sample_array_less_34()
            pass
            

    def calculate_constants_mode(self):
        for x in range(self.nTbW):
            iidx = ((x + 1)*self.intraPredAngle) >> 5
            ifact = ((x + 1)*self.intraPredAngle) & 31
            #print("When x = " + str(x) + ", f = ((" + str(x) + " + 1)* " + str(angle) + ") & 31"," = ",ifact)
            self.array_iidx.append(iidx)
            self.array_ifact.append(ifact)

    def calculate_equations_mode(self):
        equations = []
        columns = []
        for x in range(self.nTbW): 
            columns.append(x)
            current_column = []
            iidx = ((x + 1)*self.intraPredAngle) >> 5
            ifact = ((x + 1)*self.intraPredAngle) & 31
            for y in range(self.nTbH):
                current_column.append("fC[" + str(ifact) + "][0]*ref[" + str(y + iidx + 0) + "] + " + "fC[" + str(ifact) +
                                        "][1]*ref[" + str(y + iidx + 1) + "] + " + "fC[" + str(ifact) + "][2]*ref[" +
                                        str(y + iidx + 2) + "] + " + "fC[" + str(ifact) + "][3]*ref[" + str(y + iidx + 3) + "]")        
            self.equations.append(current_column)

        df = pd.DataFrame(list(zip(*self.equations)),columns = columns)
        excel_writer = pd.ExcelWriter(path + "equations_" + str(self.predModeIntra) + "_" + str(self.nTbW) + "x" + str(self.nTbH) + ".xlsx", engine='xlsxwriter') 
        df.to_excel(excel_writer, sheet_name='equations', index=False, na_rep='NaN')

        # Auto-adjust columns' width
        for column in df:
            column_width = 70
            col_iidx = df.columns.get_loc(column)
            excel_writer.sheets['equations'].set_column(col_iidx, col_iidx, column_width)

        excel_writer._save()
        return equations

    '''def calculate_equation_with_reuse(self, buffer, x):
        columns.append(x)
        current_column = []
        iidx = ((x + 1)*self.intraPredAngle) >> 5
        ifact = ((x + 1)*self.intraPredAngle) & 31
        for y in range(self.nTbH):
            if(ifact in buffer):
                if((y + iidx) in buffer[ifact]):
                    current_column.append("reuso: " + str(buffer[ifact][y + iidx]))   
                else:
                    current_column.append("fC[" + str(ifact) + "][0]*ref[" + str(y + iidx + 0) + "] + " + "fC[" + str(ifact) +
                                            "][1]*ref[" + str(y + iidx + 1) + "] + " + "fC[" + str(ifact) + "][2]*ref[" +
                                            str(y + iidx + 2) + "] + " + "fC[" + str(ifact) + "][3]*ref[" + str(y + iidx + 3) + "]")
                    buffer[ifact][y + iidx] = str(self.predModeIntra) + " : " + str(x)
            else:
                current_column.append("fC[" + str(ifact) + "][0]*ref[" + str(y + iidx + 0) + "] + " + "fC[" + str(ifact) +
                                            "][1]*ref[" + str(y + iidx + 1) + "] + " + "fC[" + str(ifact) + "][2]*ref[" +
                                            str(y + iidx + 2) + "] + " + "fC[" + str(ifact) + "][3]*ref[" + str(y + iidx + 3) + "]")
                buffer[ifact][y + iidx] = str(self.predModeIntra) + " : " + str(x) + "," + str(y)

            self.equations_reuse.append(current_column)'''
    
    
    def print_reference_sample_array(self):
        ref = []
        for i in self.ref:
            ref.append((i, self.ref[i]))

        sorted_ref = sorted(ref)
        for i in sorted_ref:
            print(i)

    def transform_dict_to_array(self, begin, end, normalize):
        ref = []
        for i in range(begin, end + 1):
            last_iidx = (((self.nTbW - 1) + 1)*self.intraPredAngle) >> 5 #iidx da última posição de x ou y
            if(last_iidx < 0):
                if(not(normalize)):
                    if(i < last_iidx):
                        ref.append("NU")
                    else:
                        ref.append(self.ref[i])
                else:
                    ref.append(self.ref[i])
            else:
                if(not(normalize)):
                    if(i > (last_iidx + (self.nTbW - 1) + 3)):
                        ref.append("NU")
                    else:
                        ref.append(self.ref[i])
                else:
                    ref.append(self.ref[i])
        return ref
            

    def normalize_ref(self):
        for i in self.ref.keys():
            x,y = re.findall(r'\d+', self.ref[i]) #Get x and y value from string
            if(i < 0):
                if(y != str(abs(i) - 1)):
                    self.ref[int('-' + str(int(y) + 1))] = self.ref[i]
                    self.ref[i] = 'NU'


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
    
def calculate_MCM_blocks(mode, state_iidx, state_ifact, base = 0):
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
            
            constant_vectors[index] = constant_vectors[index] + downward_constants[i]

        downward_index = downward_index + 1   

    for i,j in zip(constant_vectors.values(),constant_vectors.keys()):
        print(j, i)

def calculate_adders(state_iidx, state_ifact, base = 0):
    pass

#calculate_states(modes1, angles1, 64, 4)
calculate_MCM_blocks(56,"0001",[8,16,24,0])
calculate_MCM_blocks(56,"0001",[8,16,24,0], 1)
calculate_MCM_blocks(44,"1000",[24,16,8,0])
                
                
        




