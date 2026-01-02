import os
import logging
import warnings
import gradio as gr
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# --- SUPPRESS WARNINGS ---
warnings.filterwarnings("ignore")
logging.getLogger("ibm_watsonx_ai").setLevel(logging.ERROR)

# --- MODEL UTILITIES ---
def get_llm(model_id, temperature, max_tokens):
    params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.TEMPERATURE: temperature,
    }
    return WatsonxLLM(
        model_id=model_id,
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=params
    )

def run_chain(model_id, temp, tokens, template_str, inputs):
    try:
        llm = get_llm(model_id, temp, tokens)
        prompt = PromptTemplate.from_template(template_str)
        chain = (
            RunnableLambda(lambda x: prompt.format(**x)) 
            | llm 
            | StrOutputParser()
        )
        return chain.invoke(inputs)
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# --- TAB HANDLERS ---
def handle_summarize(model, temp, tokens, template, text):
    if not text.strip(): return "Please provide text to summarize."
    return run_chain(model, temp, tokens, template, {"content": text})

def handle_qa(model, temp, tokens, context, question):
    if not context or not question: return "Please provide both context and a question."
    tmpl = "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    return run_chain(model, temp, tokens, tmpl, {"context": context, "question": question})

def handle_classify(model, temp, tokens, text, labels):
    if not text.strip(): return "Please provide text to classify."
    tmpl = "Classify this text into one of these labels: {labels}\n\nText: {text}\n\nLabel:"
    return run_chain(model, temp, tokens, tmpl, {"labels": labels, "text": text})

def handle_sql(model, temp, tokens, query):
    if not query.strip(): return "-- Please provide a request."
    tmpl = "Generate a SQL query for the following request: {query}\n\nSQL:"
    return run_chain(model, temp, tokens, tmpl, {"query": query})

def handle_chat(message, history, model, temp, tokens, role):
    tmpl = f"System: {role}\n\nChat History:\n{{history}}\n\nUser: {{input}}\nAssistant:"
    hist_str = "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in history if h[0] and h[1]])
    return run_chain(model, temp, tokens, tmpl, {"history": hist_str, "input": message})

# --- UI DEFINITION ---
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
).set(
    block_title_text_weight="600",
    section_header_text_weight="700"
)

with gr.Blocks(theme=theme, title="LangChain Studio") as demo:
    # Sidebar for Global Settings
    with gr.Sidebar(label="Global Settings"):
        gr.Markdown("## ⚙️ Studio Configuration")
        gr.Markdown("Adjust the model parameters for all studio modules.")
        
        model_id = gr.Dropdown(
            label="LLM Engine",
            choices=[
                ("Granite 3.0 8B", "ibm/granite-3-8b-instruct"),
                ("Llama 3.1 70B", "meta-llama/llama-3-1-70b-instruct"),
                ("Llama 3.2 11B", "meta-llama/llama-3-2-11b-vision-instruct")
            ],
            value="ibm/granite-3-8b-instruct",
            info="Select the foundational model for inference."
        )
        temp = gr.Slider(0.0, 1.0, value=0.7, label="Temperature", 
                         info="Higher values increase creativity, lower values increase focus.")
        tokens = gr.Slider(128, 2048, value=1024, step=128, label="Max Tokens",
                          info="Maximum length of the generated response.")
        
        gr.Markdown("---")
        gr.Markdown("### 🛠️ Utilities")
        clear_all = gr.Button("Reset Studio", variant="secondary")

    # Main Area
    gr.Markdown("# 🧠 Prompt Nexus")
    gr.Markdown("Explore advanced prompt engineering patterns with IBM watsonx.ai.")

    with gr.Tabs():
        # --- Summarization ---
        with gr.Tab("📝 Summarization"):
            gr.Markdown("### Create concise summaries from long-form content.")
            with gr.Row():
                with gr.Column(scale=2):
                    sum_in = gr.Textbox(label="Input Text", placeholder="Paste the text you want to summarize here...", lines=10)
                with gr.Column(scale=1):
                    sum_tmpl = gr.Textbox(
                        label="Prompt Template", 
                        value="Summarize this in clear bullet points:\n\n{content}\n\nSummary:",
                        lines=5
                    )
                    sum_btn = gr.Button("Generate Summary", variant="primary")
            
            sum_out = gr.Textbox(label="AI Generated Summary", lines=8, interactive=False)
            sum_btn.click(handle_summarize, [model_id, temp, tokens, sum_tmpl, sum_in], sum_out)

        # --- Q&A ---
        with gr.Tab("❓ Contextual Q&A"):
            gr.Markdown("### Extract specific information based on provided context.")
            with gr.Row():
                with gr.Column():
                    qa_ctx = gr.Textbox(label="Context Source", placeholder="Provide the background info or document text...", lines=8)
                    qa_q = gr.Textbox(label="Specific Question", placeholder="What would you like to know?")
                    qa_btn = gr.Button("Extract Answer", variant="primary")
                with gr.Column():
                    gr.Markdown("#### AI Response")
                    qa_out = gr.Markdown()
            
            qa_btn.click(handle_qa, [model_id, temp, tokens, qa_ctx, qa_q], qa_out)

        # --- Classification ---
        with gr.Tab("🏷️ Classification"):
            gr.Markdown("### Categorize text into predefined labels.")
            with gr.Row():
                cl_text = gr.Textbox(label="Input Text", placeholder="Text to analyze...", scale=2)
                cl_labels = gr.Textbox(label="Labels", value="Urgent, Billing, Technical, General", scale=1)
            
            cl_btn = gr.Button("Analyze Category", variant="primary")
            cl_out = gr.Label(label="Classification Result")
            
            cl_btn.click(handle_classify, [model_id, temp, tokens, cl_text, cl_labels], cl_out)

        # --- SQL ---
        with gr.Tab("🔧 SQL Generator"):
            gr.Markdown("### Transform natural language into executable SQL queries.")
            sql_in = gr.Textbox(label="Business Requirement", placeholder="e.g., Show me the top 5 customers by revenue in 2023")
            sql_btn = gr.Button("Generate SQL Code", variant="primary")
            sql_out = gr.Code(label="Generated Query", language="sql")
            
            sql_btn.click(handle_sql, [model_id, temp, tokens, sql_in], sql_out)

        # --- Chat ---
        with gr.Tab("🎭 Agent Persona"):
            gr.Markdown("### Interactive chat with a specific AI persona.")
            chat_role = gr.Textbox(label="AI Instructions / Persona", value="A helpful technical architect specialized in cloud solutions.")
            chatbot = gr.Chatbot(height=450, show_label=False)
            
            with gr.Row():
                chat_msg = gr.Textbox(label="Your Message", placeholder="Type here...", scale=4)
                chat_send = gr.Button("Send", variant="primary", scale=1)

            def user(user_message, history):
                if history is None: history = []
                return "", history + [[user_message, None]]

            def bot(history, model, temperature, max_tokens, role):
                if not history: return history
                user_message = history[-1][0]
                bot_message = handle_chat(user_message, history[:-1], model, temperature, max_tokens, role)
                history[-1][1] = bot_message
                return history

            chat_msg.submit(user, [chat_msg, chatbot], [chat_msg, chatbot], queue=False).then(
                bot, [chatbot, model_id, temp, tokens, chat_role], chatbot
            )
            chat_send.click(user, [chat_msg, chatbot], [chat_msg, chatbot], queue=False).then(
                bot, [chatbot, model_id, temp, tokens, chat_role], chatbot
            )

    gr.Markdown("---")
    gr.Markdown("© 2024 LangChain Studio | Built with IBM watsonx.ai")

if __name__ == "__main__":
    demo.launch(share=True)