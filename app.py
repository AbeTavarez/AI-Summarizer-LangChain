import streamlit as st 
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()
st.title("Summarizer App")
st.divider()

st.markdown("## Start summarizing your documents.")


# Upload file
uploaded_file = st.file_uploader("Upload a Text, PDF or CSV file.", type=["txt", "pdf", "csv"])

# llm = ChatGroq(model="mixtral-8x7b-32768")
llm = ChatOpenAI(model="gpt-4o-mini")

parser = StrOutputParser()

prompt_template = ChatPromptTemplate.from_template("Summarize the following document {document}")

# Chain
chain = prompt_template | llm | parser


if uploaded_file is not None:
    with st.spinner("Processing..."):
        try:
            print("File: ", uploaded_file)
            print("File Type: ", uploaded_file.type)
            
            temp_file_path = uploaded_file.name
            print("File path: ", temp_file_path)
            
            # Save uploaded file
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # Create document loader
            if uploaded_file.type == "text/plain":
                loader = TextLoader(temp_file_path)
            elif uploaded_file.type == "text/csv":
                loader = CSVLoader(temp_file_path)
            elif uploaded_file.type == "application/pdf":
                loader = PyPDFLoader(temp_file_path)
            else:   
                st.error("File type is not supported!")
                st.stop()
            
            # Create the document
            doc = loader.load()
            print(doc)
            
            
            # Text Splitter
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            
            # Document Chunks   
            chunks = text_splitter.split_documents(doc)
            print(chunks)
            
        except Exception as e:
            print(e)
    
    st.success("File Uploaded")
    
    
    
    
# Summarize
if st.button("Summarize"):
    container = st.empty()
    chunk_summaries = []
    # Summarize Chunks
    with st.spinner("Summarizing Chunks"):
        try:
           for i, chunk in enumerate(chunks):
               print(f"Processing chunk {i + 1}/{len(chunks)}")
               
               # prompt
               chunk_prompt = ChatPromptTemplate.from_template(
                   "You are a highly skilled AI model tasked with summarizing text. "
                    "Please summarize the following chunk of text in a concise manner, "
                    "highlighting the most critical information. Do not omit any key details:\n\n{document}"
               ) 
               
               # chain
               chunk_chain = chunk_prompt | llm | parser
               chunk_summary = chunk_chain.invoke({"document": chunk})
               chunk_summaries.append(chunk_summary)
        
        except Exception as e:
            print("Error summarizing chunks", e)
            st.error(f"Error summarizing chunks: {e}")
            st.stop()
    # print("CHUNKS SUMMARIES ", chunk_summaries)
    
    
    
    # Final Summary
    with st.spinner("Creating final summary..."):
        try:
            # Combine all summaries
            combined_summaries = "\n".join(chunk_summaries)
            
            # Final summary prompt
            final_prompt = ChatPromptTemplate.from_template(
                "You are an expert summarizer tasked with creating a final summary from summarized chunks. "
                "Combine the key points from the provided summaries into a cohesive and comprehensive summary. "
                "The final summary should be concise but detailed enough to capture the main ideas:\n\n{document}"
            )
            
            final_chain = final_prompt | llm | parser 
            final_summary = final_chain.invoke({"document": combined_summaries})
            
            print("FINAL SUMMARY", final_summary)
            container.write(final_summary)
            
            # Download file
            st.download_button(
                label="Download Final Summary",
                data=final_summary,
                file_name="final_summary.txt",
                mime="text/plain"
            )
            
        except Exception as e:
            print("Error creating final summary", e)
            st.error(f"Error creating final summary {e}")