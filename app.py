import gradio as gr
import torch
from model import GPT, GPTConfig  # You'll need to upload your model definition
import tiktoken

# Load the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GPT(GPTConfig())
model.load_state_dict(torch.load('best_model.pt', map_location=device))
model.to(device)
model.eval()

enc = tiktoken.get_encoding('gpt2')

def generate_text(prompt, max_length=100, temperature=0.7):
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(x)[0]
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)
            
            if next_token.item() == enc.encode('\n')[0]:
                break
    
    generated_text = enc.decode(x[0].tolist())
    return generated_text

interface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(lines=2, label="Prompt"),
        gr.Slider(minimum=10, maximum=200, value=100, label="Max Length"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, label="Temperature")
    ],
    outputs=gr.Textbox(lines=5, label="Generated Text"),
    title="Shakespeare Text Generator",
    description="Enter a prompt to generate Shakespeare-style text"
)

interface.launch() 