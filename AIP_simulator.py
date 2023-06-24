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
        int refIdx
        int refW
        int refH
        int cIdx
        int refFilterFlag
        int{ref_index} ref
        int[] ref_id
        int[] array_iFact
        int[] array_iIdx
    '''

    
    def __init__(self, nTbW, nTbH, predModeIntra, intraPredAngle, refIdx, refW, refH, cIdx):
        #Initialize inputs
        self.nTbW = nTbW
        self.nTbH = nTbH
        self.predModeIntra = predModeIntra
        self.intraPredAngle = intraPredAngle
        self.refIdx = refIdx
        self.refW = refW
        self.refH = refH
        self.cIdx = cIdx
        

        #refFilterFlag hardcoded for now
        self.refFilterFlag = 1

        #Initialize reference as an empty list
        self.ref = defaultdict(lambda: "Null")
        self.ref_id = []

        #Initialize iFact and iIdx array as empty lists
        self.array_iFact = []
        self.array_iIdx = []

        self.equations = []
        self.equations_reuse = []

    def calculate_reference_sample_array_greather_equal_34(self):
        index_x = 0
        index_y = - 1 - self.refIdx

        #with x = 0...nTbW + refIdx + 1. The +1 on the end is to include (nTbW + refIdx + 1) in the array
        for x in range((self.nTbW + self.refIdx + 1) + 1):
            #ref[x] = p[-1 -refIdx + x][-1 -refIdx] 
            index_x = -1 - self.refIdx + x
            self.ref[x] = "p[" + str(index_x) + "][" + str(index_y) + "]"
            self.ref_id.append(x)

        if (self.intraPredAngle < 0):
            invAngle = round((512*32)/self.intraPredAngle)                
            index_x = - 1 - self.refIdx
            index_y = 0

            #with x = -nTbH ... -1
            for x in range(-self.nTbH, 0):
                #ref[x] = p[-1 -refIdx][-1 -refIdx + Min((x*invAngle + 256) >> 9, nTbH)]
                index_y = -1 - self.refIdx + min((x*invAngle + 256) >> 9, self.nTbH)
                self.ref[x] = "p[" + str(index_x) + "][" + str(index_y) + "]"
                self.ref_id.append(x)
                    
        else:
            index_y = - 1 - self.refIdx
            #with x = nTbW + 2 + refIdx ... refW + refIdx
            for x in range(self.nTbW + 2 + self.refIdx, (self.refW + self.refIdx) + 1):
                #ref[x] = p[-1 -refIdx + x][-1 -refIdx]
                index_x = -1 - self.refIdx + x
                self.ref[x] = ("p[" + str(index_x) + "][" + str(index_y) + "]")
                self.ref_id.append(x)
            
            index_x = -1 + self.refW
            #with x = 1...(Max(1,nTbW/nTbH)*refIdx + 1)
            for x in range(1,(max(1,self.nTbW/self.nTbH)*self.refIdx + 1) + 1):
                #ref[refW + refIdx + x] = p[-1 -refW][-1 -refIdx]
                self.ref[self.refW + self.refIdx + x] = "p[" + str(index_x) + "][" + str(index_y) + "]"
                self.ref_id.append(x)

        self.ref_id.sort()

    def calculate_reference_sample_array_less_34(self):
        index_x = - 1 - self.refIdx
        index_y = 0

        #with x = 0...nTbH + refIdx + 1. The +1 on the end is to include (nTbH + refIdx + 1) in the array
        for x in range((self.nTbH + self.refIdx + 1) + 1):
            #ref[x] = p[-1 -refIdx][-1 -refIdx + x] 
            index_y = -1 - self.refIdx + x
            self.ref[x] = "p[" + str(index_x) + "][" + str(index_y) + "]"
            self.ref_id.append(x)

        if (self.intraPredAngle < 0):
            invAngle = round((512*32)/self.intraPredAngle)                
            index_x = 0
            index_y = - 1 - self.refIdx

            #with x = -nTbW ... -1
            for x in range(-self.nTbW, 0):
                #ref[x] = p[-1 -refIdx + Min((x*invAngle + 256) >> 9, nTbW][-1 -refIdx]
                index_x = -1 - self.refIdx + min((x*invAngle + 256) >> 9, self.nTbH)
                self.ref[x] = "p[" + str(index_x) + "][" + str(index_y) + "]"
                self.ref_id.append(x)
                    
        else:
            index_x = - 1 - self.refIdx
            #with x = nTbH + 2 + refIdx ... refH + refIdx
            for x in range(self.nTbH + 2 + self.refIdx, (self.refH + self.refIdx) + 1):
                #ref[x] = p[-1 -refIdx][-1 -refIdx + x]
                index_y = -1 - self.refIdx + x
                self.ref[x] = ("p[" + str(index_x) + "][" + str(index_y) + "]")
                self.ref_id.append(x)
            
            index_y = -1 + self.refH
            #with x = 1...(Max(1,nTbH/nTbW)*refIdx + 1)
            for x in range(1,(max(1,self.nTbH/self.nTbW)*self.refIdx + 1) + 1):
                #ref[refH + refIdx + x] = p[-1 -refIdx][-1 -refH]
                self.ref[self.refH + self.refIdx + x] = "p[" + str(index_x) + "][" + str(index_y) + "]"
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
            iIdx = ((x + 1)*self.intraPredAngle) >> 5
            iFact = ((x + 1)*self.intraPredAngle) & 31
            #print("When x = " + str(x) + ", f = ((" + str(x) + " + 1)* " + str(angle) + ") & 31"," = ",iFact)
            self.array_iIdx.append(iIdx)
            self.array_iFact.append(iFact)

    def calculate_equations_mode(self):
        equations = []
        columns = []
        for x in range(self.nTbW): 
            columns.append(x)
            current_column = []
            iIdx = ((x + 1)*self.intraPredAngle) >> 5
            iFact = ((x + 1)*self.intraPredAngle) & 31
            for y in range(self.nTbH):
                current_column.append("fC[" + str(iFact) + "][0]*ref[" + str(y + iIdx + 0) + "] + " + "fC[" + str(iFact) +
                                        "][1]*ref[" + str(y + iIdx + 1) + "] + " + "fC[" + str(iFact) + "][2]*ref[" +
                                        str(y + iIdx + 2) + "] + " + "fC[" + str(iFact) + "][3]*ref[" + str(y + iIdx + 3) + "]")        
            self.equations.append(current_column)

        df = pd.DataFrame(list(zip(*self.equations)),columns = columns)
        excel_writer = pd.ExcelWriter(path + "equations_" + str(self.predModeIntra) + "_" + str(self.nTbW) + "x" + str(self.nTbH) + ".xlsx", engine='xlsxwriter') 
        df.to_excel(excel_writer, sheet_name='equations', index=False, na_rep='NaN')

        # Auto-adjust columns' width
        for column in df:
            column_width = 70
            col_idx = df.columns.get_loc(column)
            excel_writer.sheets['equations'].set_column(col_idx, col_idx, column_width)

        excel_writer._save()
        return equations

    def calculate_equation_with_reuse(self, buffer, x):
        columns.append(x)
        current_column = []
        iIdx = ((x + 1)*self.intraPredAngle) >> 5
        iFact = ((x + 1)*self.intraPredAngle) & 31
        for y in range(self.nTbH):
            if(iFact in buffer):
                if((y + iIdx) in buffer[iFact]):
                    current_column.append("reuso: " + str(buffer[iFact][y + iIdx]))   
                else:
                    current_column.append("fC[" + str(iFact) + "][0]*ref[" + str(y + iIdx + 0) + "] + " + "fC[" + str(iFact) +
                                            "][1]*ref[" + str(y + iIdx + 1) + "] + " + "fC[" + str(iFact) + "][2]*ref[" +
                                            str(y + iIdx + 2) + "] + " + "fC[" + str(iFact) + "][3]*ref[" + str(y + iIdx + 3) + "]")
                    buffer[iFact][y + iIdx] = str(self.predModeIntra) + " : " str(x)
            else:
                current_column.append("fC[" + str(iFact) + "][0]*ref[" + str(y + iIdx + 0) + "] + " + "fC[" + str(iFact) +
                                            "][1]*ref[" + str(y + iIdx + 1) + "] + " + "fC[" + str(iFact) + "][2]*ref[" +
                                            str(y + iIdx + 2) + "] + " + "fC[" + str(iFact) + "][3]*ref[" + str(y + iIdx + 3) + "]")
                buffer[iFact][y + iIdx] = str(self.predModeIntra) + " : " str(x) + "," + str(y)

            self.equations_reuse.append(current_column)
    
    
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
            last_iIdx = (((self.nTbW - 1) + 1)*self.intraPredAngle) >> 5 #iIdx da última posição de x ou y
            if(last_iIdx < 0):
                if(not(normalize)):
                    if(i < last_iIdx):
                        ref.append("NU")
                    else:
                        ref.append(self.ref[i])
                else:
                    ref.append(self.ref[i])
            else:
                if(not(normalize)):
                    if(i > (last_iIdx + (self.nTbW - 1) + 3)):
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

def calculate__angles(modes, angles, size):
    values_ifact = []
    values_idx = []
    columns = []
    for i,j in zip(modes,angles):
        tb = TransformBlock(size, size, i, j, 0, size*2 + 2, size*2 + 2, 0)
        tb.calculate_constants_mode()
        columns.append(i)
        values_idx.append(tb.array_iIdx)
        values_ifact.append(tb.array_iFact)
       
    df = pd.DataFrame(list(zip(*values_idx)),columns = columns)
    df.to_excel(excel_writer = path + "values_idx_" + str(size) + ".xlsx")
    df = pd.DataFrame(list(zip(*values_ifact)),columns = columns)
    df.to_excel(excel_writer = path + "values_ifact_" + str(size) + ".xlsx")

    

def write_samples_angles(modes, angles, size, normalize):
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


def calculate_equations(modes, angles,size):
    for i,j in zip(modes,angles):
        tb = TransformBlock(size, size, i, j, 0, size*2 + 2, size*2 + 2, 0)
        tb.calculate_equations_mode()

def calculate_all_equations_reuse(modes, angles,size):
    blocks = []
    for i,j in zip(modes,angles):
        blocks.append(TransformBlock(size, size, i, j, 0, size*2 + 2, size*2 + 2, 0))

    for tb in blocks:
        tb.calculate_equations_mode()




