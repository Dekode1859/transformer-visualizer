from transformers import AutoTokenizer, AutoModel, utils
from bertviz import model_view
utils.logging.set_verbosity_error()  # Suppress standard warnings

def get_predictions(input_text):
    model_name = "microsoft/xtremedistil-l12-h384-uncased"
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model(inputs)
    attention = outputs[-1]
    tokens = tokenizer.convert_ids_to_tokens(inputs[0])
    model_html = model_view(attention, tokens, html_action="return")
    with open("static/model_view.html", 'w') as file:
        file.write(model_html.data)
    
