import streamlit as st
import PyPDF2 as pp
from transformers import pipeline,AutoTokenizer

def extract(pdf):
    text=""
    with pdf as f:
        read=pp.PdfReader(f)
        pages=len(read.pages)
        for pagen in range(pages):
            page=read.pages[pagen]
            text+=page.extract_text()
    return text

#Test function Replace it with the BERT MODEL
def classify(text):
    tokenizer=AutoTokenizer.from_pretrained("knowledgator/comprehend_it-base")
    classifier=pipeline("zero-shot-classification",model="knowledgator/comprehend_it-base",from_pt=True)
    labels=["Education Material","Creative Writing","Government Document","Bussiness Document","Technical Document","Legal Document","Financial Document","News Document","Medical Document","Scientific Reserach Papers"]
    result=classifier(text,labels)
    top_results = sorted(zip(result['scores'], result['labels'], range(len(labels))), reverse=True)[:3]
    return top_results
def main():
    st.title('Document Classifier')
    row1,row2=st.columns([2,1])
    with row1:
        st.subheader('Extracted Text')
        text_output=st.empty()
    with row2:
        st.subheader('Classification Result')
        classification_output=st.empty()
    with st.expander("Upload PDF",expanded=True):
        up=st.file_uploader("Upload a PDF file",type="pdf")
        if up is not None:
            if st.button('Run'):
                ex=extract(up)
                text_output.text_area("Extracted Text",value=ex,height=500)
                #testfucntion
                classification_result = classify(ex)
                classification_output.write(f"Classification Result: {classification_result}")

               
if __name__=='__main__':
    main()
