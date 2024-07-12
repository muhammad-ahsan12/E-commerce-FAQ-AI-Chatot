# E-commerce FAQ Retrieval System🛒

This project implements an FAQ retrieval system for an e-commerce platform using Streamlit and various LangChain components. The system allows users to ask questions related to e-commerce and retrieves relevant answers from a pre-loaded FAQ dataset.

## Features

- **Streamlit Interface:** A user-friendly web interface for interacting with the FAQ retrieval system.
- **Language Model Integration:** Utilizes Google's Gemini-pro model for generating answers.
- **Vector Database:** Stores FAQ data in a FAISS vector database for efficient retrieval.
- **Custom Prompt:** Employs a custom prompt template to ensure accurate and relevant responses.

## Installation

1. **Clone the repository:**

    git clone [https://github.com/muhammad-ahsan12/ecommerce-faq-retrieval.git](https://github.com/muhammad-ahsan12/Ecomerse-Chatbot.git)
    cd ecommerce-faq-retrieval
    ```

2. **Install the required packages:**

    pip install -r requirements.txt

3. **Set up the Google API key:**

    Replace `YOUR_GOOGLE_API_KEY` with your actual Google API key in the code:

    os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"
   

## Usage

1. **Run the Streamlit app:**

    streamlit run app.py
   

2. **Interact with the interface:**

    - Enter your question in the input box.
    - Click on the "🔮Get Answer" button to retrieve the answer.n


# **Streamlit Interface:**
  
    st.title("E-commerce FAQ Retrieval System🛒")
    query = st.text_input("Enter your question:")
    if st.button("🔮Get Answer"):
        with st.spinner("🔄processing..."):
            result = faq_chain(query)
            st.write(result)

## Dataset

The model is trained on a dataset of 200 Question-Answer pairs specific to e-commerce. You can explore the dataset [here](https://huggingface.co/datasets/MakTek/Customer_support_faqs_dataset).

## Notes

- The model is specifically trained for e-commerce purposes.
- It may not accurately answer questions outside the scope of the training data.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.

## License

This project is licensed under the MIT License.

## Contact

For any inquiries, please contact [muhammadahsanuetm143@gmail.com].

---

You can find this project on GitHub [here](https://github.com/muhammad-ahsan12/Ecomerse-Chatbot.git).
