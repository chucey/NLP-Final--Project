import gradio as gr
from rag_retrival import load_vectorstore, retrieve_reviews_for_summary
from prompt import load_model, summarize_reviews

print("Loading model once at startup...")
tok, model = load_model(model_name="Qwen/Qwen3-0.6B")

def generate_review_summary(
    business_name=None,
    city=None,
    state=None,
    categories=None,
    review_stars=None,
):
    metadata_filter = {
        "business_name": business_name,
        "city": city,
        "state": state,
        "categories": categories,
        "review_stars": review_stars,
    }

    metadata_filter = {
        key: value for key, value in metadata_filter.items() if value is not None
    }

    rag = load_vectorstore()
    docs = retrieve_reviews_for_summary(rag, metadata_filter=metadata_filter, k=10)
    summary = summarize_reviews(docs=docs, model=model, tok=tok)
    return summary

def reset_filters():
    return None, None, None, None, None, ""

css = """
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    font-family: Arial, sans-serif;
}
.container {
    max-width: 950px;
    margin: auto;
    padding-top: 40px;
}
.header {
    text-align: center;
    color: white;
    margin-bottom: 25px;
}
.title {
    font-size: 2.4rem;
    font-weight: 700;
}
.subtitle {
    opacity: 0.8;
    margin-top: 6px;
}
.card {
    background: var(--block-background-fill);
    color: var(--body-text-color);
    border-radius: 18px;
    padding: 10px;
    margin-bottom: 12px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
"""

with gr.Blocks() as demo:
    with gr.Column(elem_classes="container"):
        gr.HTML("""
        <div class="header">
            <div class="title">Yelp Review Analyzer</div>
            <div class="subtitle">Summarize customer feedback using your RAG pipeline</div>
        </div>
        """)

        with gr.Group(elem_classes="card"):
            gr.Markdown("## Filters")
            with gr.Row():
                business_name = gr.Textbox(label="Business Name", placeholder="e.g. Joe's Pizza")
                city = gr.Textbox(label="City", placeholder="e.g. New Orleans")
                state = gr.Textbox(label="State", placeholder="e.g. PA")
                categories = gr.Textbox(label="Categories", placeholder="e.g. Italian, Pizza")
                stars =  gr.Dropdown(
                                label="Review Stars (optional)",
                                choices=[("Any", None), 1, 2, 3, 4, 5],
                                value=None,
                                allow_custom_value=False,
                            )

            with gr.Row():
                reset_btn = gr.Button("Reset", variant="secondary")
                run_btn = gr.Button("Analyze", variant="primary")

        with gr.Group(elem_classes="card"):
            gr.Markdown("## Summary")
            summary_output = gr.Textbox(label="", lines=10)

    run_btn.click(
        fn=generate_review_summary,
        inputs=[business_name, city, state, categories, stars],
        outputs=summary_output,
    )

    reset_btn.click(
        fn=reset_filters,
        inputs=[],
        outputs=[business_name, city, state, categories, stars, summary_output],
    )

demo.launch(inbrowser=True,
            css=css,)
