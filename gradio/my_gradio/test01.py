import gradio as gr

def greet(name):
    return "Hello! " + name + "!!"

iface = gr.Interface(fn=greet, 
                     inputs=gr.Textbox(lines=2, placeholder='이름을입력하세요'),
                     outputs='text')
iface.launch()