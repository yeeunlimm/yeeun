import gradio as gr

def flip_text(x):
    return x[::-1]

def greet(name):
    return "Hello " + name + "!!"

with gr.Blocks() as demo:
    gr.Markdown("인사하기 또는 글자뒤집기")
    with gr.Tab("인사하기"):
        text_input = gr.Textbox(label='이름을 입력하세요')
        text_output = gr.Textbox(label="인사")
        text_button = gr.Button("인사하기")
    with gr.Tab("뒤집기"):
        flip_input = gr.Textbox()
        flip_output = gr.Textbox()
        flip_button = gr.Button("뒤집기")

    text_button.click(greet, inputs=text_input, outputs=text_output)
    flip_button.click(flip_text, inputs=flip_input, outputs=flip_output)

demo.launch()
