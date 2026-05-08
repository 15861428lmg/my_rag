import os
import gradio as gr
from rag_engine import RAGEngine

engine = RAGEngine()

def handle_file_upload(files):
    if not files:
        return "请先选择文件"
    
    file_paths = []
    for file in files:
        file_paths.append(file.path if hasattr(file, 'path') else file.name)
    
    result = engine.add_documents(file_paths)
    return result

def handle_chat(message, history):
    if not message.strip():
        return history, ""
    
    if history is None:
        history = []
    
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})
    
    for partial_response in engine.query_stream(message):
        history[-1]["content"] = partial_response
        yield history, ""

def handle_clear_memory():
    engine.clear_memory()
    return []

def handle_clear_knowledge():
    engine.clear_vectorstore()
    engine.clear_memory()
    return "知识库已清空"

with gr.Blocks(title="智能文档问答系统") as app:
    gr.Markdown("# 智能文档问答系统")
    gr.Markdown("基于RAG技术，支持上传文档作为知识库，具有对话记忆能力")
    
    with gr.Tab("对话"):
        chatbot = gr.Chatbot(label="对话记录", height=400)
        with gr.Row():
            msg = gr.Textbox(label="输入问题", placeholder="请输入您的问题...", scale=4)
            submit_btn = gr.Button("发送", scale=1, variant="primary")
        clear_btn = gr.Button("清空对话记忆")
        
        msg.submit(handle_chat, inputs=[msg, chatbot], outputs=[chatbot, msg])
        submit_btn.click(handle_chat, inputs=[msg, chatbot], outputs=[chatbot, msg])
        clear_btn.click(handle_clear_memory, outputs=[chatbot])
    
    with gr.Tab("知识库管理"):
        file_output = gr.Textbox(label="上传结果")
        file_input = gr.File(label="上传文档", file_count="multiple", 
                           file_types=[".pdf", ".docx", ".txt"])
        upload_btn = gr.Button("上传并处理文档", variant="primary")
        clear_kb_btn = gr.Button("清空知识库", variant="stop")
        
        upload_btn.click(handle_file_upload, inputs=[file_input], 
                        outputs=[file_output])
        clear_kb_btn.click(handle_clear_knowledge, 
                          outputs=[file_output])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
