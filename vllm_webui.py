import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
import io
import json
from dotenv import load_dotenv
import base64
import numpy as np
import os
from langchain.tools import Tool
import pandas as pd
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from mcp_tools.rag_client import RAG_SEARCH
load_dotenv()

rag_client = RAG_SEARCH()

recommand_prompt = """你是一位经验丰富的智能招聘顾问。
用户的提问是：query: {query}
你可以参考的岗位信息如下：
reference: {reference}

请完成以下任务：
首先，根据 query 精准筛选匹配的岗位，如果 reference 中有完全或高度相关的岗位，请优先推荐这些岗位，并简洁说明推荐理由。
如果没有匹配的岗位，请以热情友好、推销式的语气，推荐其它你认为合适的岗位，强调岗位的优势、潜力或适配度，让用户对这些岗位产生兴趣。
输出时请尽量自然、有吸引力，不要机械列出岗位，而是像与用户对话一样，有引导性和说服力。
保持内容简洁明了，但不失人情味。
最终输出请以推荐语句的形式给出，不需要解释你的推荐逻辑。"""

tool = [
    {
        "type": "function",
        "function": {
            "name": "get_job_info",
            "description": "从本地知识库获取岗位招聘信息。可以根据地点、职位、经验级别、所需技能和公司名称进行筛选。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "招聘岗位所在的城市或地区，例如 '北京', '上海', '远程'",
                    },
                    "job_title": {
                        "type": "string",
                        "description": "岗位的具体名称或关键词，例如 '软件工程师', '数据科学家', '前端开发'",
                    },
                    "experience_level": {
                        "type": "string",
                        "description": "岗位所需的经验级别，例如 '初级', '中级', '高级', '资深'",
                        "enum": ["不限", "初级", "中级", "高级", "资深", "实习生"]
                    },
                    "skills": {
                        "type": "array",
                        "description": "岗位所需的核心技能列表，例如 ['Python', 'SQL', '机器学习', 'Java']",
                        "items": {"type": "string"}
                    },
                    "company_name": {
                        "type": "string",
                        "description": "特定公司的名称，例如 '字节跳动', '腾讯', '阿里巴巴'",
                    },
                },
                "required": ["location"],
            },
        },
    }
]

llm = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"), api_key=os.environ.get("OPENAI_API_KEY"))

class ModelChat():
    def __init__(self):
        self.history = [] # for gradio
        self.messages = [] # for llm

    def __call__(self, query: str, history: list):
        history.append({"role": "user", "content": query})

        if len(self.messages) == 0:
            self.messages = history.copy()
        else:
            self.messages.append({"role": "user", "content": query})

        response = llm.chat.completions.create(
            model=os.environ.get("MODEL_NAME"),
            messages=self.messages,
            temperature=0.7,
            tools=tool,
            tool_choice="auto",
        ).choices[0]

        if response.message.content is not None:
            history.append({"role": "assistant", "content": response.message.content})
            self.messages.append({"role": "assistant", "content": response.message.content})

        if response.finish_reason == "tool_calls":
            tool_call = response.message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            tool_result =  rag_client(" ".join([str(value) for value in tool_args.values()]))

            self.messages[-1] = {
                "role": "user",
                "content": recommand_prompt.format(
                    query = history[-1]['content'],
                    reference = tool_result
                )
            }
            
            response = llm.chat.completions.create(
                model=os.environ.get("MODEL_NAME"),
                messages=self.messages,
                temperature=0.7,
                tools=tool,
                tool_choice="none",
            ).choices[0]

            history.append({"role": "assistant", "content": response.message.content})
            self.messages.append({"role": "assistant", "content": response.message.content})

        return "", history

modelchat = ModelChat()

# 静态数据 - 简历解析结果
static_resume_data = {
    "user_name": "张三",
    "user_education": "XX大学 计算机科学 硕士",
    "user_experience": "3年深度学习工程师经验，负责模型设计与优化。",
    "user_skills": "Python, 深度学习, 自然语言处理, PyTorch, Transformer, 机器学习, Git, SQL"
}

# 静态数据 - 职位推荐列表
static_jobs_data = {
    "职位名称": [
        "高级AI算法工程师", "自然语言处理专家", "推荐系统开发工程师", "数据科学家", "机器学习研究员"
    ],
    "公司": [
        "大厂科技", "创新智能", "未来数据", "数智互联", "前沿AI"
    ],
    "地点": [
        "北京", "上海", "深圳", "杭州", "北京"
    ],
    "薪资 (K/月)": [
        "25-45", "30-50", "20-40", "28-48", "35-60"
    ],
    "匹配度": [
        0.92, 0.87, 0.85, 0.83, 0.80
    ],
    "职位ID": [
        "J001", "J002", "J003", "J004", "J005"
    ]
}

# 静态数据 - 推荐理由
static_reason = """
**高级AI算法工程师** (大厂科技)：

该职位高度匹配您的**3年深度学习工程师经验，负责模型设计与优化**以及精通的**Python, 深度学习, 自然语言处理, PyTorch**技能。我们从知识库中检索到，该职位要求扎实的**深度学习**和**自然语言处理**背景，与您的技能栈高度吻合。此外，该职位所在公司在**人工智能**领域具有领先地位，与您的职业发展方向契合。
"""

# 静态数据 - 行业洞察
static_industry_insight = """
### 行业洞察：高级AI算法工程师领域

近期**《2025人工智能人才发展报告》**指出，**高级AI算法工程师**相关领域正处于高速发展期，特别是**大模型微调与应用开发**需求激增。数据显示，未来三年该领域人才缺口预计达到**30%**。建议您进一步关注**多模态大模型**和**AI安全**以增强竞争力。

#### 薪资趋势
- 一线城市平均年薪：70-120万
- 核心技能溢价：拥有Transformer、多模态经验者薪资上浮20-30%

#### 热门公司
1. 大厂科技 - 专注于通用人工智能研发
2. 创新智能 - 垂直领域大模型应用
3. 未来数据 - 金融AI解决方案提供商
"""

# 生成技能雷达图（静态）
def create_skill_radar_chart():
    user_skills = ["Python", "深度学习", "自然语言处理", "PyTorch", "Transformer", "机器学习"]
    job_title = "高级AI算法工程师"
    
    num_vars = len(user_skills)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    # 用户技能得分
    user_values = [0.9, 0.85, 0.8, 0.75, 0.7, 0.85]
    user_values += user_values[:1]

    # 职位所需技能得分
    job_values = [0.8, 0.9, 0.7, 0.8, 0.9, 0.8]
    job_values += job_values[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, user_values, color='red', alpha=0.25, label='您的技能熟练度')
    ax.plot(angles, user_values, color='red', linewidth=2)

    ax.fill(angles, job_values, color='blue', alpha=0.25, label=f'{job_title}所需技能')
    ax.plot(angles, job_values, color='blue', linewidth=2)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(user_skills)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title(f'技能匹配雷达图 - {job_title}', va='bottom')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"

# 生成静态职位推荐数据框
def get_static_jobs_df():
    df = pd.DataFrame(static_jobs_data)
    return df

# 生成静态技能雷达图
def get_static_radar_chart():
    img_data = create_skill_radar_chart()
    buf = io.BytesIO(base64.b64decode(img_data.split(',')[1]))
    img = Image.open(buf)
    return img

# --- Gradio 界面设计 ---

with gr.Blocks(theme=gr.themes.Soft(), title="RAG就业推荐系统") as demo:
    gr.Markdown(
        """
        # 🚀 基于RAG的智能就业推荐系统
        本系统结合了**检索增强生成 (RAG)** 技术与深度学习模型，为您提供**个性化、可解释**的职位推荐。
        通过上传简历和填写求职偏好，系统将从海量知识库中检索并生成最适合您的职位及匹配理由。
        """
    )
    
    # 系统状态提示
    status_output = gr.Textbox(
        label="系统状态",
        value="已加载演示数据",
        interactive=False,
        show_copy_button=True,
        elem_id="status_box",
        lines=1
    )

    with gr.Tabs():
        with gr.TabItem("简历上传与解析"):
            with gr.Row():
                with gr.Column(scale=1):
                    resume_file = gr.File(label="上传您的简历 (PDF/TXT)", file_types=["pdf", "txt"])
                    parse_resume_btn = gr.Button("解析简历", variant="primary")
                    
                with gr.Column(scale=2):
                    gr.Markdown("#### 简历解析结果")
                    user_name = gr.Textbox(label="姓名", value=static_resume_data["user_name"], interactive=False)
                    user_education = gr.Textbox(label="教育背景", value=static_resume_data["user_education"], interactive=False)
                    user_experience = gr.Textbox(label="工作经验", value=static_resume_data["user_experience"], interactive=False)
                    user_skills = gr.Textbox(label="技能标签", value=static_resume_data["user_skills"], interactive=False)
            
            # 按钮点击事件 - 仅更新状态
            parse_resume_btn.click(
                fn=lambda: "简历解析成功！（演示模式）",
                inputs=None,
                outputs=status_output
            )

        with gr.TabItem("求职偏好与推荐"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 您的求职偏好")
                    desired_job_title = gr.Textbox(label="期望职位关键词", placeholder="例如：AI工程师, NLP研究员", value="AI算法工程师")
                    desired_location = gr.Dropdown(
                        ["北京", "上海", "深圳", "杭州", "广州", "全国"], 
                        label="期望工作地点", 
                        value="北京"
                    )
                    desired_salary_min = gr.Slider(
                        minimum=5, maximum=100, step=1, value=20, label="期望月薪 (K) - 最小值"
                    )
                    desired_salary_max = gr.Slider(
                        minimum=5, maximum=100, step=1, value=40, label="期望月薪 (K) - 最大值"
                    )
                    
                    recommend_btn = gr.Button("获取个性化推荐", variant="primary")
                    
                with gr.Column(scale=2):
                    gr.Markdown("#### 职位推荐列表")
                    recommended_jobs_df = gr.DataFrame(
                        headers=["职位名称", "公司", "地点", "薪资 (K/月)", "匹配度"],
                        datatype=["str", "str", "str", "str", "number"],
                        label="推荐职位",
                        interactive=False
                    )
                    
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 推荐理由 (针对第一条推荐)")
                    recommended_reason = gr.Markdown(static_reason)
                    
                with gr.Column():
                    gr.Markdown("#### 技能匹配度分析")
                    skill_radar_plot = gr.Image(label="技能雷达图", value=get_static_radar_chart())

            # 推荐按钮点击事件 - 更新推荐结果
            recommend_btn.click(
                fn=lambda: (get_static_jobs_df(), static_reason, "推荐成功！（演示模式）", get_static_radar_chart()),
                inputs=None,
                outputs=[recommended_jobs_df, recommended_reason, status_output, skill_radar_plot]
            )
            
            # 当推荐列表被点击时，更新推荐理由和技能雷达图
            def update_selected_job_info(evt: gr.SelectData):
                if evt.index is not None:
                    # 根据点击的行索引更新推荐理由和雷达图
                    # 这里简化处理，仅展示不同的职位名称
                    job_title = static_jobs_data["职位名称"][evt.index[0]]
                    new_reason = static_reason.replace("高级AI算法工程师", job_title)
                    return new_reason, get_static_radar_chart()
                return gr.No(), gr.No()

            recommended_jobs_df.select(
                fn=update_selected_job_info,
                outputs=[recommended_reason, skill_radar_plot]
            )

        with gr.TabItem("行业洞察"):
            gr.Markdown("#### 与您期望职位相关的最新行业洞察与趋势分析")
            industry_insight = gr.Markdown(static_industry_insight)
            
            # 推荐按钮点击时，更新行业洞察
            recommend_btn.click(
                fn=lambda: static_industry_insight,
                inputs=None,
                outputs=industry_insight
            )

        with gr.TabItem("求职推荐系统"):
            gr.Markdown("#### 查询您适合什么工作")
            with gr.Column(scale=10):
                chatbot = gr.Chatbot(
                    label="聊天区域",
                    height=400,
                    type="messages",
                    value=[{"role": "system", "content": "你是一个提供岗位推荐的智能助手。当用户询问某个岗位时，你首先尝试匹配你已知的岗位数据。如果你发现自己库中没有关于用户所提岗位的具体信息，那么请以模糊、概括的方式回答，并主动推荐你所了解的与此领域相关的其他岗位数据。例如，当用户查询一个你没有直接记录的岗位时，你可以回答：“关于这个岗位的信息我这里资料比较有限，不过我注意到在这个领域有一些类似的职位，比如……你可以了解一下这些岗位。”请务必注意，回答时保持语气友好且具有建设性，帮助用户获得更多方向的信息。当用户问题超出你知识范围或需要查询最新数据时，请主动调用工具检索相关信息，并用工具返回的内容生成回答。"}]
                )
                msg = gr.Textbox(label="输入框", placeholder="请输入你的问题...")
                with gr.Row():
                    submit = gr.Button("发送")
                    clear = gr.ClearButton([msg, chatbot])
            
        # 添加事件处理
        msg.submit(modelchat, [msg, chatbot], [msg, chatbot])
        submit.click(modelchat, [msg, chatbot], [msg, chatbot])

    # 底部版权信息
    gr.Markdown(
        """
        ---
        © 2025 RAG就业推荐系统. All rights reserved.
        """
    )

    # 设置初始数据
    demo.load(
        fn=lambda: (
            get_static_jobs_df(),
            static_reason,
            get_static_radar_chart()
        ),
        inputs=None,
        outputs=[recommended_jobs_df, recommended_reason, skill_radar_plot]
    )

if __name__ == "__main__":
    demo.queue() # 启用队列，处理并发请求
    demo.launch(debug=True, inline=False, share=False, allowed_paths=["."]) # inline=False会在浏览器中打开新标签页

