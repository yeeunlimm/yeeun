import gradio as gr

def greet(name, is_morning, temp):
    hello = "Good Morning" if is_morning else "Good Evening"
    greeting = f"{hello} {name}! It is {temp} degrees today"
    celsius = (temp - 32)*5 / 9
    return greeting, round(celsius, 2)

iface = gr.Interface(
                    fn = greet,
                    inputs=['text','checkbox', gr.Slider(0,100)],
                    outputs=['text', 'number'])
iface.launch()