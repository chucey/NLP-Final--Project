# imports
import gradio as gr
from rag_retrival import load_vectorstore, retrieve_reviews_for_summary
from prompt import load_model, summarize_reviews

def gradio_output(business_name: str = None,
                  city: str = None, 
                  state: str = None, 
                  categories: str = None, 
                  review_stars: int = None) -> str:

    if review_stars in ("", None):
        review_stars = None
    else:
        review_stars = int(review_stars)


    # ----Create the metadatafilter based on user inputs. Only include non-None values.----
    metadata_filter = {
        "business_name": business_name,
        "city": city,
        "state": state,
        "categories": categories,
        "review_stars": review_stars}

    metadata_filter = {
        key: value for key, value in metadata_filter.items() if value is not None
    }

    # --- Load RAG vectorstore ---
    rag = load_vectorstore()

    # --- Retrieve relevant reviews based on metadata filter ---
    docs = retrieve_reviews_for_summary(rag, metadata_filter=metadata_filter, k=10)

    # --- Load the language model ---
    tok, model = load_model(model_name="Qwen/Qwen3-0.6B")

    # --- Generate summary from retrieved reviews ---
    summary = summarize_reviews(docs=docs, model=model, tok=tok)

    return summary

with gr.Blocks() as demo:
    gr.Markdown("## Yelp Review Summarizer")
    gr.Markdown("Enter metadata criteria to retrieve and summarize Yelp reviews.")
    
    with gr.Row():
        business_name = gr.Textbox(label="Business Name", placeholder="e.g. Joe's Pizza")
        city = gr.Textbox(label="City", placeholder="e.g. New York")
        state = gr.Textbox(label="State", placeholder="e.g. NY")
        categories = gr.Textbox(label="Categories", placeholder="e.g. Italian, Pizza")
        review_stars = gr.Dropdown(
            label="Review Stars (optional)",
            choices=[("Any", None), 1, 2, 3, 4, 5],
            value=None,
            allow_custom_value=False,
        )
    with gr.Row():    
        with gr.Column():
                clear_button = gr.ClearButton(
                        components=[business_name, city, state, categories, review_stars],
                        value="Reset Filters",)    
        with gr.Column():             
                summarize_button = gr.Button("Summarize Reviews", variant="primary")
   
    output_area = gr.Textbox(label="Summary Output", lines=10)

    summarize_button.click(
        fn=gradio_output,
        inputs=[business_name, city, state, categories, review_stars],
        outputs=output_area
    )

if __name__ == "__main__":
    demo.launch()