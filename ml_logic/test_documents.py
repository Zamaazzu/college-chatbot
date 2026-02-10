from document_retriever import find_most_relevant_document

query = "When is the exam"
doc_name, content = find_most_relevant_document(query)

print("Most Relevant Document:", doc_name)
print("Preview:\n")
print(content[:500])
