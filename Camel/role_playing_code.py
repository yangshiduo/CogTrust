from colorama import Fore
import os
from typing import List
from camel.agents.chat_agent import FunctionCallingRecord
from camel.societies import RolePlaying
from camel.types import TaskType, ModelType, ModelPlatformType, RoleType
from camel.configs import ChatGPTConfig
from camel.models import ModelFactory
import numpy as np
from camel.agents import TrustManAgent,chat_agent
from camel.messages import BaseMessage
from camel.generators import SystemMessageGenerator
from camel.prompts.base import TextPrompt
import randomly_perturb_prompts
import pandas as pd
from camel.toolkits.code_execution import CodeExecutionToolkit

toolkit = CodeExecutionToolkit(verbose=False)
tools = toolkit.get_tools()

OPENAI_API_KEY = ''
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

def extract_executed_results(text):
    # 找到 "Executed Results:" 的起始位置
    start_index = text.find("Embodied Actions:")

    # 如果找到了，就截取 "Executed Results:" 之后的内容
    if start_index != -1:
        return text[start_index + len("Embodied Actions:"):].strip()

    # 如果没有找到，返回一个提示
    return "No 'Executed Results:' found in the text."

# 定义声誉更新函数
def calculate_exponential_decay_weights(lambda_value, n):
    """
    计算每轮声誉值的记忆因子，使用指数衰减公式 m_i = e^(-lambda * i)
    :param lambda_value: 衰减系数
    :param n: 需要考虑的历史轮数
    :return: 记忆因子数组
    """
    weights = np.exp(-lambda_value * np.arange(n))
    return weights

def update_reputation(T_history, lambda_value):
    """
    计算更新后的声誉值 T''，基于历史声誉值 T 和指数衰减记忆因子
    :param T_history: 历史声誉值列表，最近的声誉值排在最前面
    :param lambda_value: 衰减系数
    :return: 更新后的声誉值 T''
    """
    n = len(T_history)  # 历史声誉值的个数
    weights = calculate_exponential_decay_weights(lambda_value, n)  # 计算记忆因子
    T_double_prime = np.dot(weights, T_history) / np.sum(weights)  # 加权平均计算新的声誉值
    return T_double_prime


def write_to_excel(file_name, value, col_index):
    # 将数据逐列写入Excel文件
    df = pd.DataFrame({f'col{col_index}': [value]})

    with pd.ExcelWriter(file_name, mode='a', engine='openpyxl') as writer:
        # 确定写入的位置
        startrow = writer.sheets['Sheet1'].max_row
        # 将值写入指定列
        for index, row in df.iterrows():
            writer.sheets['Sheet1'].cell(row=startrow + index + 1, column=col_index + 1, value=row[f'col{col_index}'])

def main(task_prompt_file, row_index, model=None, chat_turn_limit=10) -> None:
# Initial the role-playing session on developing a trading bot task with default model (`GPT_4O_MINI`)
    print("task:", task_prompt_file)
    output_directory = r"E:\ysd\CAMEL_test\role_play_result_GPT-3.5_safe_tool"
    os.makedirs(output_directory, exist_ok=True)  # 确保目录存在
    file_name = f"task_{row_index + 32}.xlsx"  # +2 因为索引从0开始，第二行是1
    file_path = os.path.join(output_directory, file_name)
    df = pd.DataFrame(columns=['user_generate', 'assistant_action', 'T_user', 'T_assistant', 'Trust_user', 'Trust_assistant', 'extracted_values', 'task_prompt'])
    df.to_excel(file_path, index=False)# 创建一个新的Excel文件
    task_prompt = task_prompt_file

    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_3_5_TURBO,
        model_config_dict=ChatGPTConfig().as_dict(),
    )
    role_play_session = RolePlaying(
        assistant_role_name="Python Programmer",
        assistant_agent_kwargs=dict(model=model, tools=tools,),
        user_role_name="computer expert",
        user_agent_kwargs=dict(model=model),
        task_prompt=task_prompt,
        with_task_specify=False,
        #task_specify_agent_kwargs=dict(model=model),
    )

# Output initial message with different colors.
    print(
        Fore.GREEN
        + f"AI Assistant sys message:\n{role_play_session.assistant_sys_msg}\n"
    )
    print(
        Fore.BLUE + f"AI User sys message:\n{role_play_session.user_sys_msg}\n"
    )

    print(Fore.YELLOW + f"Original task prompt:\n{task_prompt}\n")
    print(
        Fore.CYAN
        + "Specified task prompt:"
        + f"\n{role_play_session.specified_task_prompt}\n"
    )
    print(Fore.RED + f"Final task prompt:\n{role_play_session.task_prompt}\n")
    write_to_excel(file_path, role_play_session.task_prompt, 7)

    n = 0
    input_msg = role_play_session.init_chat()
    #定义日志
    society_log = "===== ai_society_log =====\n"
    society_log += "user_capability: \n" + role_play_session.user_sys_msg.content + "\n"+"---end_user_capability---\n"
    society_log += "assistant_capability: \n" + role_play_session.assistant_sys_msg.content + "\n"+"---end_assistant_capability:---\n"
    society_log += "expected_results: \n" + role_play_session.task_prompt + "\n"+"---end_expected_results---\n"
    # 定义TrustManAgent类 返回信任等级
    role_name = "Trust Manager"
    meta_dict = dict(role=role_name, task="Answer questions")
    sys_msg = SystemMessageGenerator().from_dict(
        meta_dict=meta_dict, role_tuple=(role_name, RoleType.TRUST_MAN))
    Trust_Man_Agent = TrustManAgent(
        system_message = sys_msg,
        model = model,
        #model_config = model_config
    )
    # 初始化变量
    lambda_value = 0.5  # 记忆因子衰减系数
    T_assistant_history = []  # 用来记录历史声誉值
    T_user_history = []
    #max_iterations = 10  # 最大迭代次数
    max_history_length = 5  # 历史记录最大长度
    #current_iteration = 0
    Trust_user = 1
    Trust_assistant = 1

# Output response step by step with different colors.
# Keep output until detect the terminate content or reach the loop limit.
    while n < chat_turn_limit:
        n += 1
        society_log_trust = "===== ai_society_log =====\n"
        society_log_trust += "user_capability: \n" + role_play_session.user_sys_msg.content + "\n" + "---end_user_capability---\n"
        society_log_trust += "assistant_capability: \n" + role_play_session.assistant_sys_msg.content + "\n" + "---end_assistant_capability:---\n"
        #assistant_response, user_response = role_play_session.step(input_msg, Trust_user, Trust_assistant)
        assistant_response, user_response = role_play_session.step(input_msg)

        if assistant_response.terminated:
            print(
                Fore.GREEN
                + (
                    "AI Assistant terminated. Reason: "
                    f"{assistant_response.info['termination_reasons']}."
                )
            )
            break
        if user_response.terminated:
            print(
                Fore.GREEN
                + (
                    "AI User terminated. "
                    f"Reason: {user_response.info['termination_reasons']}."
                )
            )
            break

        print(
            Fore.BLUE + f"AI User:\n\n{user_response.msg.content}\n"
        )
        print(Fore.GREEN + "AI Assistant:")
        tool_calls: List[FunctionCallingRecord] = [
            FunctionCallingRecord(**call.as_dict())
            for call in assistant_response.info['tool_calls']
        ]
        for func_record in tool_calls:
            print(f"{func_record}")
        print(f"{assistant_response.msg.content}\n")

        # 回复的内容添加到log里
        society_log += "user_generate: \n" + user_response.msg.content + "\n" + "---end_user_generate---\n"
        society_log += "assistant_action: \n" + assistant_response.msg.content + "\n" + "---end_assistant_action---\n"
        society_log_trust += "user_generate: \n" + user_response.msg.content + "\n" + "---end_user_generate---\n"
        society_log_trust += "assistant_action: \n" + assistant_response.msg.content + "\n" + "---end_assistant_action---\n"

        write_to_excel(file_path, user_response.msg.content, 0)
        write_to_excel(file_path, assistant_response.msg.content, 1)

        # 创建一个BaseMessage对象作为input_message传递给trust_evaluate
        trust_input_message = BaseMessage(
            role_name="Trust Manager",
            role_type=RoleType.TRUST_MAN,
            meta_dict=None,
            content=""
        )
        T_user, T_assistant, extracted_values = Trust_Man_Agent.trust_evaluate(society_log=society_log_trust,
                                                             input_message=trust_input_message)

        write_to_excel(file_path, T_user, 2)
        write_to_excel(file_path, T_assistant, 3)

        # 长期意见+短期意见
        # 将当前声誉值添加到历史记录
        T_user_history.insert(0, T_user)  # 把最新的值放在列表最前面
        T_assistant_history.insert(0, T_assistant)

        # 检查历史记录是否超过最大长度
        if len(T_user_history) > max_history_length:
            T_user_history.pop()  # 删除最老的声誉值（即最后一个元素）
        if len(T_assistant_history) > max_history_length:
            T_assistant_history.pop()  # 删除最老的声誉值（即最后一个元素）

        # 基于历史声誉值和记忆因子更新声誉值
        Trust_user = update_reputation(T_user_history, lambda_value)
        Trust_assistant = update_reputation(T_assistant_history, lambda_value)

        write_to_excel(file_path, Trust_user, 4)
        write_to_excel(file_path, Trust_assistant, 5)
        write_to_excel(file_path, extracted_values, 6)
        print("Trust_user:", Trust_user)
        print("Trust_assistant:", Trust_assistant)

        if "CAMEL_TASK_DONE" in user_response.msg.content:
            break
        #对消息进行随机扰动
        '''perturb_type = randomly_perturb_prompts.random_insert_updated
        input_msg_perturb = randomly_perturb_prompts.role_playing_perturb(Trust_assistant, assistant_response.msg.content, perturb_type)
        print("input_msg_perturb: ", input_msg_perturb)
        assistant_response.msg.content = input_msg_perturb'''
        input_msg = assistant_response.msg

if __name__ == "__main__":
    # 读取Excel文件中的第一列任务
    excel_file = r"E:\ysd\CAMEL_test\data_tiny.xlsx"
    df = pd.read_excel(excel_file)

    # 获取第一列，从第二行开始
    tasks = df.iloc[30:60, 0].dropna().tolist()  # 跳过第一行，并且跳过空值

    # 依次处理每一个任务
    for index, task_prompt_file in enumerate(tasks):
        main(task_prompt_file, index)