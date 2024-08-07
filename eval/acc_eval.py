def acc(predict,ground):
    flag_predict = 2
    flag_ground = 3
    negative_list = ["不","不是","否"]

    for word in negative_list:
        if word in predict:
            flag_predict = 0
        if word in ground:
            flag_ground = 0
    
    if flag_ground != 0 and "是" in ground:
        flag_ground = 1
    
    if flag_predict != 0 and "是" in predict:
        flag_predict = 1
    
    #对特殊情况的处理，因为 ground里面数字后面总有点
    ground.strip(".")
    
    # 倘若有一方不包含这样的判断词，或者是 ground 字数偏多可能是陈述句则维持原样
    if flag_predict == 2 or flag_ground == 3 or len(ground) > 2:
        if ground in predict:
            return True
        else:
            return False
    else: # 对于判断问题进行单独判断
        return flag_ground == flag_predict
