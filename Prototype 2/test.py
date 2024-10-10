from langchain.prompts import PromptTemplate

def rephrase_query(query):
    print(query)
    
    rephrase_prompt = PromptTemplate.from_template("""
    You are an expert in rephrasing legal queries. Please rephrase the following query in a formal legal manner:
    Query: {query}
    Rephrased Query:
    """)
    
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    response = rephrase_prompt.format(query=query)
    print(f"Rephrased Query: {response}")
    
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


rephrase_query("Namaste")