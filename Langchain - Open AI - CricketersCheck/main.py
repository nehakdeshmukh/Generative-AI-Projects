import os
from constant import openai_key 
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st

os.environ["OPENAI_API_KEY"]= openai_key

st.title("CricketersCheck")

input_text = st.text_input("Search the Cricketer's name")


#### prompt templates

first_input_prompt =PromptTemplate(
    input_variables=['name'],
    template="Tell me about cricketer {name}"
    )


# Memory 
person_memory = ConversationBufferMemory(input_key = 'name', memory_key="chat_history")
match_memory = ConversationBufferMemory(input_key = 'person', memory_key="chat_history")
win_memory = ConversationBufferMemory(input_key = 'name', memory_key="chat_history")


# Open AI Model and llm chains 
llm = OpenAI(temperature=0.8)
chain1 = LLMChain(llm=llm, prompt=first_input_prompt,verbose=True,output_key='person_name',memory=person_memory)


#### prompt templates

second_input_prompt =PromptTemplate(
    input_variables=['person_name'],
    template="Tell me the number of matches {person_name} played"
    )

chain2 = LLMChain(llm=llm, prompt=first_input_prompt,verbose=True,output_key="matches_count",memory=match_memory)

#### prompt templates

Third_input_prompt =PromptTemplate(
    input_variables=['matches_count'],
    template="from total {matches_count} the number matches win by him"
    )

chain3 = LLMChain(llm=llm, prompt=first_input_prompt,verbose=True,output_key="win_count",memory=win_memory)



parent_chain = SequentialChain(chains=[chain1,chain2,chain3], input_variables=["name"],
                               output_variables=["person_name","matches_count","win_count"],verbose=True)

if input_text:
    st.write(parent_chain({"name":input_text}))

