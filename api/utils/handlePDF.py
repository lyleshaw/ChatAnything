import pickle

from PyPDF2 import PdfReader

from ..utils.embedding import create_embeddings, QA


def extract_pdf(file_embedding: str, file_path: str):
    reader = PdfReader(file_path)
    print("total pages ", len(reader.pages))
    texts = []
    for i in range(len(reader.pages)):
        page = reader.pages[i]
        text = page.extract_text()
        text = text.split("\n")
        texts += text
    texts = [text.strip() for text in texts if text.strip()]
    print(texts)
    data_embedding, tokens = create_embeddings(texts)
    pickle.dump(data_embedding, open(file_embedding, 'wb'))
    print("文本消耗 {} tokens".format(tokens))


def ask(file_embedding: str, question: str):
    data_embedding = pickle.load(open(file_embedding, 'rb'))
    qa = QA(data_embedding)
    answer, context = qa(question)
    return answer, context


if __name__ == '__main__':
    extract_pdf("t.pkl", "1.pdf")