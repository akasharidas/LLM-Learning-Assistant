import argparse
import glob
import hashlib
import os
import pickle

from dotenv import load_dotenv
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain, RetrievalQA
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers.audio import OpenAIWhisperParser, OpenAIWhisperParserLocal
from langchain.document_loaders.youtube import _parse_video_id
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled, YouTubeTranscriptApi

ctx_lengths = {
    "gpt-3.5-turbo": 4000,
    "gpt-3.5-turbo-16k": 16000,
    "gpt-4": 8000,
    "gpt-4-32k": 32000,
    "llama2": 4000
}

def cache_path(url):
    """Return a suitable cache path for the given URL."""
    return os.path.join("./cache", hashlib.md5(url.encode()).hexdigest() + ".pkl")

def colored_print(text, color_code="\033[94m"):
    """Print text using the provided ANSI color code."""
    print(f"{color_code}{text}\033[0m")

def get_transcript_directly(url):
    video_id = _parse_video_id(url)
    if video_id is None:
        return []

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    except TranscriptsDisabled:
        return []

    try:
        transcript = transcript_list.find_manually_created_transcript(["en"])
    except NoTranscriptFound:
        return []
    
    transcript_pieces = transcript.fetch()

    transcript = " ".join([t["text"].strip(" ").replace("\n", " ") for t in transcript_pieces])

    return [Document(page_content=transcript)]

def load_and_transcribe_audio(urls, save_dir, whisper_model):
    url = urls[0]
    cached_file = cache_path(url)
    
    # Check if we have a cached version of the transcript
    if os.path.exists(cached_file):
        print(f"Found cached transcript. Loading from {cached_file}...")
        with open(cached_file, 'rb') as file:
            return pickle.load(file)
    
    # If not, load and transcribe, then cache
    docs = get_transcript_directly(url)
    if len(docs)==0:
        print("Could not get transcript from API. Using Whisper to transcribe audio...")
        if whisper_model == "cloud":
            loader = GenericLoader(YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParser())
        else:
            loader = GenericLoader(YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParserLocal(lang_model=f"openai/whisper-{whisper_model}"))
        docs = loader.load()

    # Cache the result
    os.makedirs("./cache", exist_ok=True)
    with open(cached_file, 'wb') as file:
        pickle.dump(docs, file)

    return docs


def load_language_model(model_choice):
    if model_choice.startswith("gpt"):
        return ChatOpenAI(model=model_choice)
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_texts(splits, embeddings)
    return vectordb


def load_summarize_chain_stuff(llm):
    prompt_template = """
    Take a deep breath and work on this step-by-step. 
    You are a helpful academic assistant. You will be given a long document. 
    Write a detailed summary of it such that someone reading the summary can understand all the main points and takeaways of the full document.

    The summary must include the following elements:
        * A title that accurately reflects the content of the text.
        * An introduction paragraph that provides an overview of the topic.
        * A detailed summary of the main points and takeaways of the the text, in bullet points.
        * A conclusion paragraph that summarizes the main points of the text.

    DOCUMENT: "{text}"

    SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="text", verbose=True
    )

    return stuff_chain


def load_summarize_chain_mapreduce(llm, token_max):
    # Map
    map_template = """
    Take a deep breath and work on this step-by-step. 
    You are a helpful academic assistant. You will be given a document. 
    Write a detailed summary of it such that someone reading the summary can understand all the main points and takeaways of the full document.
    
    DOCUMENT: "{docs}"

    SUMMARY:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce
    reduce_template='''
    Generate a summary of the following text that includes the following elements:

    * A title that accurately reflects the content of the text.
    * An introduction paragraph that provides an overview of the topic.
    * A detailed summary of the main points and takeaways of the the text, in bullet points.
    * A conclusion paragraph that summarizes the main points of the text.

    Text:`{doc_summaries}`
    '''
    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="doc_summaries"
    )

    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain,
        collapse_documents_chain=combine_documents_chain,
        token_max=token_max,
    )

    map_reduce_chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain,
        document_variable_name="docs",
        return_intermediate_steps=False,
        verbose=True
    )

    return map_reduce_chain


def main():
    print("\n--- Long-Form Content Summary and Q&A ---")

    parser = argparse.ArgumentParser(description="Long-Form Content Summary and Q&A")
    parser.add_argument("--url", help="YouTube video URL")
    parser.add_argument("--whisper_model", default="cloud", choices=["cloud", "small", "medium", "large"], help="Variant of the Whisper model to use")
    parser.add_argument("--language_model", default="gpt-3.5-turbo", choices=["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k", "llama2"], help="Language model to use")

    args = parser.parse_args()
    
    args.ctx_length = ctx_lengths[args.language_model]

    # Print the command-line arguments
    print("\n--- Arguments ---")
    for arg_name, arg_value in vars(args).items():
        print(f"{arg_name}: {arg_value}")
    print("\n--------------------------------------\n")


    if args.whisper_model=="cloud" or args.language_model.startswith("gpt"):
        load_dotenv()

    # Directory to save audio files
    save_dir = "./downloads"
    for f in glob.glob(save_dir + "/*"):
        os.remove(f)

    colored_print("\n\nLoading and transcribing audio...")
    docs = load_and_transcribe_audio([args.url], save_dir, args.whisper_model)

    colored_print("\n\nLoading language model...")
    llm = load_language_model(args.language_model)

    colored_print("\n\nEmbedding and indexing text...")
    vectordb = embed_and_index_text(docs)

    colored_print("\n\nRunning summarization chain...")
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=args.ctx_length, chunk_overlap=0
    )
    docs = text_splitter.transform_documents(docs)

    if len(docs)>1:
        summarize_chain = load_summarize_chain_mapreduce(llm, args.ctx_length)
    else:
        summarize_chain = load_summarize_chain_stuff(llm)

    summary = summarize_chain.run(docs)
    summary = f"SUMMARY: {summary}"
    colored_print("\n\n" + summary, "\033[92m")  # print summary in green

    colored_print("\n\nSetting up the QA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
    )
    colored_print("Initialization complete. You can now ask questions!\n")

    while True:
        query = input("Enter your question (type 'quit' to exit):\n> ")
        if query.lower() == "quit":
            colored_print("\nExiting. Goodbye!\n")
            break
        else:
            colored_print("\nFetching your answer...\n")
            response = qa_chain.run(query)
            colored_print(f"Answer: {response}\n", "\033[92m")  # print answer in green

if __name__ == "__main__":
    main()
