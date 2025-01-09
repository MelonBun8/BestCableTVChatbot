import os
os.environ['USER_AGENT'] = 'myagent'

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# loaders for all the pages on the website

loader_general_bundle = WebBaseLoader("https://bestcabletv.com/")
loader_att_bundle = WebBaseLoader("https://bestcabletv.com/att/")
loader_frontier_bundle = WebBaseLoader("https://bestcabletv.com/frontier/")
loader_optimum_bundle = WebBaseLoader("https://bestcabletv.com/optimum/")
loader_spectrum_bundle = WebBaseLoader("https://bestcabletv.com/spectrum/")
loader_verizon_bundle = WebBaseLoader("https://bestcabletv.com/verizon/")
loader_windstream_bundle = WebBaseLoader("https://bestcabletv.com/windstream/")

loader_bestcabletv_tv = WebBaseLoader("https://bestcabletv.com/tv")
loader_optimum_tv = WebBaseLoader("https://bestcabletv.com/optimum/tv")
loader_spectrum_tv = WebBaseLoader("https://bestcabletv.com/spectrum/cabletv")  
loader_direct_tv = WebBaseLoader("https://bestcabletv.com/att/directv")
loader_dish_tv = WebBaseLoader("https://bestcabletv.com/frontier/dishtv")

loader_fiber_internet_type = WebBaseLoader("https://bestcabletv.com/blog/fiber-internet-providers")
loader_cable_internet_type = WebBaseLoader("https://bestcabletv.com/blog/cable-internet-providers")
loader_ipbb_internet_type = WebBaseLoader("https://bestcabletv.com/blog/ipbb-internet")
loader_dsl_internet_type = WebBaseLoader("https://bestcabletv.com/blog/dsl-internet-providers")
loader_five_g_internet_type = WebBaseLoader("https://bestcabletv.com/blog/5g-home-internet-providers")
loader_fixed_wireless_internet_type = WebBaseLoader("https://bestcabletv.com/blog/fixed-wireless-internet-providers")
loader_satellite_internet_type = WebBaseLoader("https://bestcabletv.com/blog/satellite-internet-providers")

loader_general_internet = WebBaseLoader("https://bestcabletv.com/internet-providers")
loader_att_internet = WebBaseLoader("https://bestcabletv.com/att/internet")
loader_earthlink_internet = WebBaseLoader("https://bestcabletv.com/earthlink-internet/")
loader_frontier_internet = WebBaseLoader("https://bestcabletv.com/frontier/internet")
loader_hughesnet_internet = WebBaseLoader("https://bestcabletv.com/hughesnet/internet")
loader_optimum_internet = WebBaseLoader("https://bestcabletv.com/optimum/internet")
loader_spectrum_internet = WebBaseLoader("https://bestcabletv.com/spectrum/internet")
loader_verizon_internet = WebBaseLoader("https://bestcabletv.com/verizon/internet")
loader_viasat_internet = WebBaseLoader("https://bestcabletv.com/viasat/internet")
loader_windstream_internet = WebBaseLoader("https://bestcabletv.com/windstream/internet")


#loading up the HTML content for all the pages on the website
# Bundle data
general_bundle_data = loader_general_bundle.load()
att_bundle_data = loader_att_bundle.load()
frontier_bundle_data = loader_frontier_bundle.load()
optimum_bundle_data = loader_optimum_bundle.load()
spectrum_bundle_data = loader_spectrum_bundle.load()
verizon_bundle_data = loader_verizon_bundle.load()
windstream_bundle_data = loader_windstream_bundle.load()

# TV data
bestcabletv_tv_data = loader_bestcabletv_tv.load()
optimum_tv_data = loader_optimum_tv.load()
spectrum_tv_data = loader_spectrum_tv.load()
direct_tv_data = loader_direct_tv.load()
dish_tv_data = loader_dish_tv.load()

# Internet type data
fiber_internet_type_data = loader_fiber_internet_type.load()
cable_internet_type_data = loader_cable_internet_type.load()
ipbb_internet_type_data = loader_ipbb_internet_type.load()
dsl_internet_type_data = loader_dsl_internet_type.load()
five_g_internet_type_data = loader_five_g_internet_type.load()  # Corrected 5g to five_g
fixed_wireless_internet_type_data = loader_fixed_wireless_internet_type.load()
satellite_internet_type_data = loader_satellite_internet_type.load()

# # Internet data
general_internet_data = loader_general_internet.load()
att_internet_data = loader_att_internet.load()
earthlink_internet_data = loader_earthlink_internet.load()
frontier_internet_data = loader_frontier_internet.load()
hughesnet_internet_data = loader_hughesnet_internet.load()
optimum_internet_data = loader_optimum_internet.load()
spectrum_internet_data = loader_spectrum_internet.load()
verizon_internet_data = loader_verizon_internet.load()
viasat_internet_data = loader_viasat_internet.load()
windstream_internet_data = loader_windstream_internet.load()




# Now making splits for each page
# Bundle splits
general_bundle_splits = text_splitter.split_documents(general_bundle_data)
att_bundle_splits = text_splitter.split_documents(att_bundle_data)
frontier_bundle_splits = text_splitter.split_documents(frontier_bundle_data)
optimum_bundle_splits = text_splitter.split_documents(optimum_bundle_data)
spectrum_bundle_splits = text_splitter.split_documents(spectrum_bundle_data)
verizon_bundle_splits = text_splitter.split_documents(verizon_bundle_data)
windstream_bundle_splits = text_splitter.split_documents(windstream_bundle_data)

# # TV splits
bestcabletv_tv_splits = text_splitter.split_documents(bestcabletv_tv_data)
optimum_tv_splits = text_splitter.split_documents(optimum_tv_data)
spectrum_tv_splits = text_splitter.split_documents(spectrum_tv_data)
direct_tv_splits = text_splitter.split_documents(direct_tv_data)
dish_tv_splits = text_splitter.split_documents(dish_tv_data)

# Internet type splits
fiber_internet_type_splits = text_splitter.split_documents(fiber_internet_type_data)
cable_internet_type_splits = text_splitter.split_documents(cable_internet_type_data)
ipbb_internet_type_splits = text_splitter.split_documents(ipbb_internet_type_data)
dsl_internet_type_splits = text_splitter.split_documents(dsl_internet_type_data)
five_g_internet_type_splits = text_splitter.split_documents(five_g_internet_type_data)
fixed_wireless_internet_type_splits = text_splitter.split_documents(fixed_wireless_internet_type_data)
satellite_internet_type_splits = text_splitter.split_documents(satellite_internet_type_data)

# # Internet splits
general_internet_splits = text_splitter.split_documents(general_internet_data)
att_internet_splits = text_splitter.split_documents(att_internet_data)
earthlink_internet_splits = text_splitter.split_documents(earthlink_internet_data)
frontier_internet_splits = text_splitter.split_documents(frontier_internet_data)
hughesnet_internet_splits = text_splitter.split_documents(hughesnet_internet_data)
optimum_internet_splits = text_splitter.split_documents(optimum_internet_data)
spectrum_internet_splits = text_splitter.split_documents(spectrum_internet_data)
verizon_internet_splits = text_splitter.split_documents(verizon_internet_data)
viasat_internet_splits = text_splitter.split_documents(viasat_internet_data)
windstream_internet_splits = text_splitter.split_documents(windstream_internet_data)


# Bundle vector databases
general_bundle_vector_db = FAISS.from_documents(documents=general_bundle_splits, embedding=embeddings)
# att_bundle_vector_db = FAISS.from_documents(documents=att_bundle_splits, embedding=embeddings) 
frontier_bundle_vector_db = FAISS.from_documents(documents=frontier_bundle_splits, embedding=embeddings)
optimum_bundle_vector_db = FAISS.from_documents(documents=optimum_bundle_splits, embedding=embeddings)
spectrum_bundle_vector_db = FAISS.from_documents(documents=spectrum_bundle_splits, embedding=embeddings)
verizon_bundle_vector_db = FAISS.from_documents(documents=verizon_bundle_splits, embedding=embeddings)
windstream_bundle_vector_db = FAISS.from_documents(documents=windstream_bundle_splits, embedding=embeddings)

# TV vector databases
bestcabletv_tv_vector_db = FAISS.from_documents(documents=bestcabletv_tv_splits, embedding=embeddings)
optimum_tv_vector_db = FAISS.from_documents(documents=optimum_tv_splits, embedding=embeddings)
spectrum_tv_vector_db = FAISS.from_documents(documents=spectrum_tv_splits, embedding=embeddings)
# direct_tv_vector_db = FAISS.from_documents(documents=direct_tv_splits, embedding=embeddings)
dish_tv_vector_db = FAISS.from_documents(documents=dish_tv_splits, embedding=embeddings)

# Internet type vector databases
fiber_internet_type_vector_db = FAISS.from_documents(documents=fiber_internet_type_splits, embedding=embeddings)
cable_internet_type_vector_db = FAISS.from_documents(documents=cable_internet_type_splits, embedding=embeddings)
ipbb_internet_type_vector_db = FAISS.from_documents(documents=ipbb_internet_type_splits, embedding=embeddings)
dsl_internet_type_vector_db = FAISS.from_documents(documents=dsl_internet_type_splits, embedding=embeddings)
five_g_internet_type_vector_db = FAISS.from_documents(documents=five_g_internet_type_splits, embedding=embeddings)
fixed_wireless_internet_type_vector_db = FAISS.from_documents(documents=fixed_wireless_internet_type_splits, embedding=embeddings)
satellite_internet_type_vector_db = FAISS.from_documents(documents=satellite_internet_type_splits, embedding=embeddings)

# # Internet vector databases
general_internet_vector_db = FAISS.from_documents(documents=general_internet_splits, embedding=embeddings)
# att_internet_vector_db = FAISS.from_documents(documents=att_internet_splits, embedding=embeddings)
earthlink_internet_vector_db = FAISS.from_documents(documents=earthlink_internet_splits, embedding=embeddings)
frontier_internet_vector_db = FAISS.from_documents(documents=frontier_internet_splits, embedding=embeddings)
hughesnet_internet_vector_db = FAISS.from_documents(documents=hughesnet_internet_splits, embedding=embeddings)
optimum_internet_vector_db = FAISS.from_documents(documents=optimum_internet_splits, embedding=embeddings)
spectrum_internet_vector_db = FAISS.from_documents(documents=spectrum_internet_splits, embedding=embeddings)
verizon_internet_vector_db = FAISS.from_documents(documents=verizon_internet_splits, embedding=embeddings)
viasat_internet_vector_db = FAISS.from_documents(documents=viasat_internet_splits, embedding=embeddings)
windstream_internet_vector_db = FAISS.from_documents(documents=windstream_internet_splits, embedding=embeddings)


#Finally, saving the vector DBs
general_bundle_vector_db.save_local("general_bundle")
# att_bundle_vector_db.save_local("att_bundle")
frontier_bundle_vector_db.save_local("frontier_bundle")
optimum_bundle_vector_db.save_local("optimum_bundle")
spectrum_bundle_vector_db.save_local("spectrum_bundle")
verizon_bundle_vector_db.save_local("verizon_bundle")
windstream_bundle_vector_db.save_local("windstream_bundle")

bestcabletv_tv_vector_db.save_local("bestcabletv_tv")
optimum_tv_vector_db.save_local("optimum_tv")
spectrum_tv_vector_db.save_local("spectrum_tv")
# direct_tv_vector_db.save_local("direct_tv")
dish_tv_vector_db.save_local("dish_tv")

fiber_internet_type_vector_db.save_local("fiber_internet_type")
cable_internet_type_vector_db.save_local("cable_internet_type")
ipbb_internet_type_vector_db.save_local("ipbb_internet_type")
dsl_internet_type_vector_db.save_local("dsl_internet_type")
five_g_internet_type_vector_db.save_local("five_g_internet_type")
fixed_wireless_internet_type_vector_db.save_local("fixed_wireless_internet_type")
satellite_internet_type_vector_db.save_local("satellite_internet_type")

general_internet_vector_db.save_local("general_internet")
# att_internet_vector_db.save_local("att_internet")
earthlink_internet_vector_db.save_local("earthlink_internet")
frontier_internet_vector_db.save_local("frontier_internet")
hughesnet_internet_vector_db.save_local("hughesnet_internet")
optimum_internet_vector_db.save_local("optimum_internet")
spectrum_internet_vector_db.save_local("spectrum_internet")
verizon_internet_vector_db.save_local("verizon_internet")
viasat_internet_vector_db.save_local("viasat_internet")
windstream_internet_vector_db.save_local("windstream_internet")