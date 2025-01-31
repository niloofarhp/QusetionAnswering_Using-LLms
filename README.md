This Project is using LLms for Quoestion Answering task.
Wh have used different LLm models and APIs and Analyzed different methods results.


One of the LLM models is HuggingFace llama that below are showing the output results of Question Answering.

'''
pdf_path = "/Users/niloofar/Documents/Projects/langchain/practitioners_guide_to_mlops_whitepaper.pdf"
question = "To whom you suggest this book?"
answer = query_pdf(pdf_path, question)

print("Answer:", answer)
'''


'''
Question: To whom you suggest this book?
Helpful Answer: This book is intended for teams who want details about what MLOps looks like in practice. It assumes familiarity with basic machine learning concepts and development and deployment practices such as CI/CD. Therefore, I would suggest this book to data scientists, machine learning engineers, and DevOps engineers who are looking to understand and implement MLOps processes and capabilities in their organizations.
'''
