import os
from llama_index import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.readers import PDFReader


def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )

    return index


pdf_path = os.path.join("data", "Canada.pdf")
canada_pdf = PDFReader().load_data(file=pdf_path)
canada_index = get_index(canada_pdf, "canada")
canada_engine = canada_index.as_query_engine()


lpp_path = os.path.join("data", "lpp.pdf")
lpp_pdf = PDFReader().load_data(file=lpp_path)
lpp_index = get_index(lpp_pdf, "lpp")
lpp_engine = lpp_index.as_query_engine()


apg_path = os.path.join("data", "apg.pdf")
apg_pdf = PDFReader().load_data(file=apg_path)
apg_index = get_index(apg_pdf, "apg")
apg_engine = apg_index.as_query_engine()

ifd_path = os.path.join("data", "impot_federal_dierct.pdf")
ifd_pdf = PDFReader().load_data(file=ifd_path)
ifd_index = get_index(ifd_pdf, "apg")
ifd_engine = ifd_index.as_query_engine()