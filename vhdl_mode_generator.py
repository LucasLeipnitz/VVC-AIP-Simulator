import AIP_simulator as sim

path = "./"

def generate_mode(mode, list_of_adders, list_of_references):
    mode_name = ""
    mode_references = ""
    mode_outputs = ""

    mode_name = "mode_" + str(mode)

    f = open(path + mode_name + ".vhd", "w")

    for i in list_of_references:
        str_ref = str(i)
        if(str_ref[0] == "-"):
            str_ref = 'n' + str_ref[1:]

        mode_references = mode_references + "ref_" + str_ref + ", "

    for i in range(len(list_of_adders)):
        mode_outputs = mode_outputs + "p" + str(i) + ", "

    write_header(f)
    f.write("\n\n")
    write_entity(f,mode_name, mode_references, mode_outputs)
    f.close()

def write_header(f):
    header = "-----------------------------------------------\nLIBRARY ieee;\nUSE ieee.std_logic_1164.all;\nUSE ieee.std_logic_signed.all;\nuse ieee.numeric_std.all;\n-----------------------------------------------"
    f.write(header)

def write_entity(f, mode_name, mode_references, mode_outputs):
    entity = "ENTITY " + mode_name + " IS\n\tPort(\n\t\t" + mode_references + ": in std_logic_vector ( 7 downto 0 );\n\t\t" + mode_outputs + ": out std_logic_vector ( 7 downto 0 )\n\t);\nEND " + mode_name
    f.write(entity)


    


