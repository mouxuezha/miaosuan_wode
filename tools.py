import numpy as np 

def select_by_type(units,key="obj_id",value=0):
    # a common tool, for selecting units according to different keys.
    units_selected = [] 
    for unit in units:
        value_candidate = unit[key]
        if type(value_candidate) == str:
            if value in value_candidate:
                # so this one is selected
                units_selected.append(unit)
        else:
            if value==value_candidate:
                # so this one is selected
                units_selected.append(unit)
        
    return units_selected

def is_stop(attacker_ID, model="now"):
        # 这个就是单纯判断一下这东西是不是停着
        if type(attacker_ID) == list:
            # which means units inputted.
            flag_is_stop = True
            for unit in attacker_ID:
                flag_is_stop = flag_is_stop and unit["stop"]
        else:
            # normal
            unit = attacker_ID
            flag_is_stop = unit["stop"]
        
        return flag_is_stop 

def is_exist(attacker_ID,**kargs):
        # check if this obj still exist.
        if "units" in kargs:
            units = kargs["units"]
        else:
            raise Exception("no input units, G.")

        flag_exist = False 
        for bop in units:
            if attacker_ID == bop["obj_id"]:
                flag_exist = flag_exist or True
        
        return flag_exist

def get_ID_list(status,color=0):
        # get iterable ID list from status or something like status.
        operators_dict = status["operators"]
        ID_list = [] 
        for operator in operators_dict:
            # filter, only my operators pass
            if operator["color"] == color:
                # my operators
                ID_list.append(operator["obj_id"])
        return ID_list

def get_bop(obj_id, **kargs):
        """Get bop in my observation based on its id."""
        # 这个实现好像有点那啥，循环里带循环的，后面看看有没有机会整个好点的。xxh0307
        if "status" in kargs:
            observation = kargs["status"]
        else:
            raise Exception("no input status, G.")
        for bop in observation["operators"]:
            if obj_id == bop["obj_id"]:
                return bop

def get_pos(attacker_ID, **kargs):
        # just found pos according to attacker_ID
        # print("get_pos: unfinished yet")
        unit0 = get_bop(attacker_ID,**kargs)
        pos_0 = unit0["cur_hex"]
        return pos_0

def hex_to_xy(self,hex):
        # 要搞向量运算来出阵形，所以还是有必要搞一些转换的东西的。
        y = round(hex / 100)
        x = hex - y *100
        xy = np.array([x,y]) 
        return xy
    
def xy_to_hex(self,xy):
        hex = 100*xy[1] + xy[0]
        hex = round(hex)
        return hex

def get_pos_average(units):
        geshu = len(units)
        pos_ave = 0 
        pos_sum = 0
        for i in range(geshu):
            # pos_ave = (pos_ave/(i+0.000001) + self.get_pos(units[i]["obj_id"])) / (i+1)
            pos_sum = pos_sum + get_pos(units[i]["obj_id"])
        
        pos_ave = round(pos_sum / geshu)
        return pos_ave