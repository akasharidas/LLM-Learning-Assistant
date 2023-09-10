import argparse
import pickle
import os
import torch
import hashlib
import glob
from dotenv import load_dotenv
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers.audio import OpenAIWhisperParser, OpenAIWhisperParserLocal
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.llms import LlamaCpp, OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain


def cache_path(url):
    """Return a suitable cache path for the given URL."""
    return os.path.join("./cache", hashlib.md5(url.encode()).hexdigest() + ".pkl")

def colored_print(text, color_code):
    """Print text using the provided ANSI color code."""
    print(f"{color_code}{text}\033[0m")

def load_and_transcribe_audio(urls, save_dir, whisper_model):
    url = urls[0]
    cached_file = cache_path(url)
    
    # Check if we have a cached version of the transcript
    if os.path.exists(cached_file):
        print(f"Found cached transcript. Loading from {cached_file}...")
        with open(cached_file, 'rb') as file:
            return pickle.load(file)
    
    # If not, load and transcribe, then cache
    if whisper_model == "cloud":
        loader = GenericLoader(YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParser())
    else:
        loader = GenericLoader(YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParserLocal(lang_model=f"openai/whisper-{whisper_model}"))
    docs = loader.load()

    # Cache the result
    with open(cached_file, 'wb') as file:
        pickle.dump(docs, file)

    return docs


def load_language_model(model_choice):
    if model_choice == "openai":
        return OpenAI()
    else:
        return LlamaCpp(
            model_path="./llama-2-13b-chat.Q4_0.gguf",
            n_ctx=4096,
            n_gpu_layers=50,
            n_threads=1,
            n_batch=1,
            temperature=0.75,
            top_p=1,
        )


def embed_and_index_text(docs):
    combined_docs = [doc.page_content for doc in docs]
    text = " ".join(combined_docs)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    splits = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_texts(splits, embeddings)
    return vectordb


def main():
    print("\n--- Podcast Summary and Q&A ---")

    parser = argparse.ArgumentParser(description="Podcast Summary and Q&A")
    parser.add_argument("--url", help="YouTube video URL")
    parser.add_argument("--whisper_model", default="cloud", choices=["cloud", "small", "medium", "large"], help="Variant of the Whisper model to use")
    parser.add_argument("--language_model", default="openai", choices=["openai", "llama2"], help="Language model to use")

    args = parser.parse_args()

    # Print the command-line arguments
    print("\n--- Arguments ---")
    for arg_name, arg_value in vars(args).items():
        print(f"{arg_name}: {arg_value}")
    print("\n--------------------------------------\n")


    if args.whisper_model=="cloud" or args.language_model=="openai":
        load_dotenv()

    # Directory to save audio files
    save_dir = "./downloads"
    for f in glob.glob(save_dir + "/*"):
        os.remove(f)

    colored_print("\n\nLoading and transcribing audio...", "\033[94m")
    docs = load_and_transcribe_audio([args.url], save_dir, args.whisper_model)

    colored_print("\n\nLoading language model...", "\033[94m")
    llm = load_language_model(args.language_model)

    colored_print("\n\nEmbedding and indexing text...", "\033[94m")
    vectordb = embed_and_index_text(docs)

    colored_print("\n\nRunning summarization chain...", "\033[94m")
    summarize_chain = load_summarize_chain(llm, chain_type="stuff")
    summary = summarize_chain.run(docs)
    summary = f"SUMMARY: {summary}"
    colored_print("\n\n" + summary, "\033[92m")  # print summary in green

    colored_print("\n\nSetting up the QA chain...", "\033[94m")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
    )
    colored_print("Initialization complete. You can now ask questions!\n", "\033[94m")

    while True:
        query = input("Enter your question (type 'quit' to exit):\n> ")
        if query.lower() == "quit":
            colored_print("\nExiting. Goodbye!\n", "\033[94m")
            break
        else:
            colored_print("\nFetching your answer...\n", "\033[94m")
            response = qa_chain.run(query)
            colored_print(f"Answer: {response}\n", "\033[92m")  # print answer in green

if __name__ == "__main__":
    main()
