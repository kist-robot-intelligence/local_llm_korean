# 오프라인 사용 가능한 llama 3 inference code
# VRAM 16GB 이상 되어야 실행 가능

# from huggingface_hub import login
# login(token='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')

# 허깅페이스 모델/토크나이저를 다운로드 받을 경로
# ./cache/ 경로에 다운로드 받도록 설정
import os
os.environ["TRANSFORMERS_CACHE"] = "/home/kistsc/Desktop/gogi/langchain/cache/"
#os.environ["HF_HOME"] = "/home/kistsc/Desktop/gogi/langchain/cache/"

# 사용할 모델의 ID를 지정합니다.
#model_id = "MLP-KTLim/llama-3-Korean-Bllossom-8B"  
model_id = "beomi/Llama-3-KoEn-8B-Instruct-preview"

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer
import torch

from langchain.memory import ConversationSummaryBufferMemory

tokenizer = AutoTokenizer.from_pretrained(
    model_id
)  # 지정된 모델의 토크나이저를 로드합니다.

tokenizer.eos_token_id = tokenizer.encode("<|eot_id|>")[0]

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

terminators = [
tokenizer.eos_token_id,
tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', torch_dtype=torch.float16)  # 지정된 모델을 로드합니다.

streamer = TextStreamer(tokenizer, skip_prompt=True)

# 텍스트 생성 파이프라인을 생성하고, 최대 생성할 새로운 토큰 수를 10으로 설정합니다.
pipe = pipeline("text-generation", model=model,
                tokenizer=tokenizer, max_new_tokens=512, pad_token_id=tokenizer.pad_token_id, repetition_penalty=1.2, eos_token_id=terminators, streamer=streamer)
# HuggingFacePipeline 객체를 생성하고, 생성된 파이프라인을 전달합니다.
hf = HuggingFacePipeline(pipeline=pipe)

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# assistant"""  # 질문과 답변 형식을 정의하는 llama3용 템플릿

template = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

친절한 챗봇으로서 상대방의 요청에 간결하고 친절하게 답하자. 모든 대답은 한국어(Korean)으로 대답해줘.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

이제까지의 너와 내가 대화한 목록이야.
대화목록:
{history}

이건 나의 요구사항이야.
요구사항:
{input}

모든 대답은 한국어(Korean)으로 대답해줘.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""  # 질문과 답변 형식을 정의하는 템플릿

summary_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
제공된 대화 내용을 점진적으로 요약하고, 이전 요약에 새로운 요약을 추가합니다. 요약은 반드시 한국어를 사용해야 합니다.<|eot_id|><|start_header_id|>user<|end_header_id|>
현재 요약:\n{summary}\n\n새로운 대화:\n{new_lines}\n\n새로운 요약:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""  # 요약 기능을 수행하는 템플릿

prompt = PromptTemplate.from_template(template)  # 템플릿을 사용하여 프롬프트 객체 생성
summary_prompt = PromptTemplate.from_template(summary_template)

# 프롬프트와 언어 모델을 연결하여 체인 생성
chain = prompt | hf | StrOutputParser()

from langchain.chains import ConversationChain

conversation = ConversationChain(
    prompt=prompt,
    llm=hf,
    memory=ConversationSummaryBufferMemory(llm=hf, max_token_limit=4000, human_prefix='user', ai_prefix='assistant', return_messages=True, prompt=summary_prompt, return_only_outputs=True),
)

conversation_chain = conversation

while 1:
    
    print('>>> ',end='')
    question = input()
    # 300자 약 900토큰으로 input 제한
    question = question[0:300]
    print('question is ',question)
    print('len is ',len(question))

    rst = conversation_chain.invoke({"input":question}, return_only_outputs=True)
    
    # print("\n\n\n[memory]")
    # print(conversation_chain.memory)
    # print("\n\n\n")