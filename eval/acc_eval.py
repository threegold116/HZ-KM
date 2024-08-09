import openai
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
    
def llm_judge(predict,ground,question):

    
    openai.api_key = "EMPTY"
    openai.base_url = "http://localhost:8022/v1/"

    model = "Qwen2-7B-Instruct"
    # prompt = "Once upon a time"

    # # create a completion
    # completion = openai.completions.create(model=model, prompt=prompt, max_tokens=64)
    # # print the completion
    # print(prompt + completion.choices[0].text)

    # create a chat completion
    # prompt_with_question='''
    # 问题: "{q}"
    # 标准答案: "{g}"
    # 模型预测的答案: "{p}"
    # 任务: 判断上述文本中模型预测的答案是否与标准答案一致？只回答“是”或“否”。  
    # '''.format(q=question,g=ground,p=predict)
    prompt_with_question='''
    问题: "{q}"
    标准答案: "{g}"
    模型预测的答案: "{p}"
    任务: 判断上述文本中模型预测的答案是否正确？如果完全符合回答“正确”，如果部分符合或完全不符合回答“错误”。只回答“正确”或“错误”。  
    '''.format(q=question,g=ground,p=predict)
    completion = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user", 
                "content": prompt_with_question
            }],
        max_tokens=256
    )
    # print the completion
    judge_result=completion.choices[0].message.content.lower()
    if "正确" in judge_result:
        return True
    elif "错误" in judge_result:
        return False
    else:
        return False    
