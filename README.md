# BestCableTV Chatbot

A chatbot designed for the BestCableTV website to intelligently handle queries about internet and cable TV packages.

---

## Overview and Features
- Intelligently answers frequently asked questions, such as:
  - "Show me the cheapest plans in Detroit"
  - "Most economical internet packages in Detroit"
- Provides detailed information about specific internet packages based on:
  - Area
  - City
  - State
  - Zip code

---

## Underlying Technology
- **LangChain**: Handles LLM logic and creates logical chains.
- **Google Gemini 1.5 Flash**: Powers the chatbot's language generation.
- **FAISS**: Creates, stores, and loads vector databases.
- **HuggingFace Embeddings**: Uses `all-MiniLM-L6-v2` for creating vector stores.
- **SQLite**: Stores internet plan data using a SQLite database.
- **Streamlit**: Provides the graphical user interface (GUI).

---

## How to Recreate and Set Up (For Linux: CPanel – FROM Absolute Scratch)

### One-Time Setup:

#### Step 1: Install Miniconda
Follow the instructions in the [official Miniconda installation guide](https://docs.anaconda.com/miniconda/install/).

#### Step 2: Create a New Virtual Environment
Run the following commands in your terminal:
```bash
conda activate
conda create -n chatbotenv python=3.12.4
conda config --append channels conda-forge
conda install langchain langchain-google-genai streamlit faiss-cpu langchain-community bs4
```

#### Step 3: Prepare the Chatbot Folder
1. If provided with a prebuilt `VECTOR_STORES` folder, move it to your desired server location. Skip to the **Regular Use** section below.
2. Otherwise, create a folder for the chatbot (e.g., `chatbotenv`) and:
   - Move all source code files, the `.env` file, `.db`, and `.sql` files into the folder.
   - Create a subfolder named `vector_stores`.

#### Step 4: Configure the `.env` File
Add your Google API key to the `.env` file:
```
GOOGLE_API_KEY='your_api_key_here'
```
You can generate an API key via [Google Cloud Console](https://console.cloud.google.com).

#### Step 5: Generate Vector Stores
Move the following scripts into the `vector_stores` folder:
- `db_creator.py`
- `CitNAreaName-Generator.py`

Run the following commands in the terminal from the `vector_stores` folder:
```bash
python3 db_creator.py
python3 CitNAreaName-Generator.py
```
**Note:** These operations may take significant time depending on the dataset size.

---

### Regular Use:
1. Open a terminal and navigate to the chatbot folder.
2. Run the following command:
   ```bash
   streamlit run main.py
   ```
3. Copy the provided **External URL** link and paste it into your browser to start using the chatbot.

---

## Credits
**Developer:** Saad Arfan

---

## License
This project is licensed under [Eplanet Inc.](#).
