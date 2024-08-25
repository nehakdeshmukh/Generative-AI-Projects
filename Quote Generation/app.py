import streamlit as st 
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

### function to get response from LLAma 3 model 
def getLLama_response(input_text,no_words,quote_style,keyword):
    
    # LLma Model 
    llm_model = CTransformers(model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
                          model_type= "llama",
                          config = {"max_new_tokens":256, "temperature":0.01})
    
    #prompt template
    template = """
                write a quote for {quote_style} occupation for a topic {input_text} 
                within {no_words} words and must include this keywords {keyword}.
                """
    
    prompt=PromptTemplate(input_variables=["quote_style","input_text","no_words","keyword"],
                      template=template) 

    response=llm_model(prompt.format(quote_style=quote_style,input_text=input_text,no_words=no_words,keyword=keyword))
    #print(response)
    return response



st.set_page_config(page_title="Instgram Quote",
                   page_icon=r"C:\\Neha\\Data Science Projects local\\model\\logo.jpeg",
                   layout="centered",
                   initial_sidebar_state="collapsed")



st.header("Generate Quote") 


st.image(r"C:\\Neha\\Data Science Projects local\\model\\logo.jpeg", width=200)

input_text = st.text_input("Enter the quote topic")

#creating 2 more details 

col1,col2,col3 = st.columns([5,5,5])


with col1:
    no_words = st.text_input("No of Words")

with col2:
    quote_style = st.selectbox("writting the quote for",("Writers and Poets","Philosophers and Thinkers","Motivational Speakers and Coaches",
                                                      "Social Media Influencers","Teachers and Educators","Business Leaders and Entrepreneurs",
                                                      "Artists and Creatives"), index=0)
    
with col3:
    keyword = st.text_input("keyword")
    
submit = st.button("Generate")
    
if submit:
    st.write(getLLama_response(input_text,no_words,quote_style,keyword))


