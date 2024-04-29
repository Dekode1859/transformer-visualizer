import streamlit as st
from transformers import AutoTokenizer, AutoModel, utils
from bertviz import model_view
import streamlit.components.v1 as components
from train import get_or_build_tokenizer, greedy_decode
from config import get_config, latest_weights_file_path
from model import build_transformer
import torch
from bertviz import model_view
import torch
import altair as alt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

utils.logging.set_verbosity_error()  # Suppress standard warnings

st.set_page_config(page_title='Attention Visualizer', layout='wide')

def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                "%.2d - %s" % (r, row_tokens[r] if len(row_tokens) > r else "<blank>"),
                "%.2d - %s" % (c, col_tokens[c] if len(col_tokens) > c else "<blank>"),
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        columns=["row", "column", "value", "row_token", "col_token"],
    )
    
    
def get_attn_map(attn_type: str, layer: int, head: int, model):
    if attn_type == "encoder":
        attn = model.encoder.layers[layer].self_attention_block.attention_scores
    elif attn_type == "decoder":
        attn = model.decoder.layers[layer].self_attention_block.attention_scores
    elif attn_type == "encoder-decoder":
        attn = model.decoder.layers[layer].cross_attention_block.attention_scores
    return attn[0, head].data

def attn_map(attn_type, layer, head, row_tokens, col_tokens, max_sentence_len, model):
    df = mtx2df(
        get_attn_map(attn_type, layer, head, model),
        max_sentence_len,
        max_sentence_len,
        row_tokens,
        col_tokens,
    )
    return (
        alt.Chart(data=df)
        .mark_rect()
        .encode(
            x=alt.X("col_token", axis=alt.Axis(title="")),
            y=alt.Y("row_token", axis=alt.Axis(title="")),
            color=alt.Color("value", scale=alt.Scale(scheme="blues")),
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        #.title(f"Layer {layer} Head {head}")
        .properties(height=200, width=200, title=f"Layer {layer} Head {head}")
        .interactive()
    )

def get_all_attention_maps(attn_type: str, layers: list[int], heads: list[int], row_tokens: list, col_tokens, max_sentence_len: int, model):
    charts = []
    for layer in layers:
        rowCharts = []
        for head in heads:
            rowCharts.append(attn_map(attn_type, layer, head, row_tokens, col_tokens, max_sentence_len, model))
        charts.append(alt.hconcat(*rowCharts))
    return alt.vconcat(*charts)

def initiate_model(config, device):
    tokenizer_src = get_or_build_tokenizer(config, None, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, None, config["lang_tgt"])

    model = build_transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config["seq_len"], config['seq_len'], d_model=config['d_model']).to(device)

    model_filename = latest_weights_file_path(config)
    state = torch.load(model_filename, map_location=torch.device('cpu'))
    model.load_state_dict(state['model_state_dict'])
    return model, tokenizer_src, tokenizer_tgt

def process_input(input_text, tokenizer_src, tokenizer_tgt, model, config, device):
    src = tokenizer_src.encode(input_text)
    src = torch.cat([
        torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64), 
        torch.tensor(src.ids, dtype=torch.int64),
        torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
        torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (config['seq_len'] - len(src.ids) - 2), dtype=torch.int64)
    ], dim=0).to(device)
    source_mask = (src != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)

    encoder_input_tokens = [tokenizer_src.id_to_token(i) for i in src.cpu().numpy()]
    encoder_input_tokens = [i for i in encoder_input_tokens if i != '[PAD]']

    model_out = greedy_decode(model, src, source_mask, tokenizer_src, tokenizer_tgt, config['seq_len'], device)

    decoder_input_tokens = [tokenizer_tgt.id_to_token(i) for i in model_out.cpu().numpy()]
    
    output = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
    
    return encoder_input_tokens, decoder_input_tokens, output
        

# def get_html_data(model_name, input_text):
#     model_name ="microsoft/xtremedistil-l12-h384-uncased"
#     model = AutoModel.from_pretrained(model_name, output_attentions=True, cache_dir='__pycache__')  # Configure model to return attention values
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     inputs = tokenizer.encode(input_text, return_tensors='pt')  # Tokenize input text
#     outputs = model(inputs)  # Run model
#     attention = outputs[-1]  # Retrieve attention from model outputs
#     tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings
#     model_html = model_view(attention, tokens, html_action="return")  # Display model view
#     with open("static/model_view.html", 'w') as file:
#         file.write(model_html.data)

def main():
    st.title('Transformer Visualizer')
    # st.info('Enter a sentence to visualize the attention of the model')
    st.write('This app visualizes the attention of a transformer model on a given sentence.')
    # add a side bar with model options and a prompt
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer_src, tokenizer_tgt = initiate_model(config, device)
    with st.sidebar:
        input_text = st.text_input('Enter a sentence')
        # put two buttons side by side in the sidebar
        # translate_button = st.button('Translate', key='translate_button')
        # viz_button = st.button('Visualize Attention', key='viz_button')
        attn_type = st.selectbox('Select attention type', ['encoder', 'decoder', 'encoder-decoder'])
        layers = st.multiselect('Select layers', list(range(6)))
        heads = st.multiselect('Select heads', list(range(8)))
        # allow the user to select the all the layers and heads at once to visualize
        if st.checkbox('Select all layers'):
            layers = list(range(6))
        if st.checkbox('Select all heads'):
            heads = list(range(8))
    
    if input_text != '':
        with st.spinner("Translating..."):
            encoder_input_tokens, decoder_input_tokens, output = process_input(input_text, tokenizer_src, tokenizer_tgt, model, config, device)
            max_sentence_len = len(encoder_input_tokens)
            row_tokens = encoder_input_tokens
            col_tokens = decoder_input_tokens
            st.write('Input:', ' '.join(encoder_input_tokens))
            st.write('Output:', ' '.join(decoder_input_tokens))
            st.write('Translated:', output)
            st.write('Attention Visualization')
        with st.spinner("Visualizing Attention..."):
            if attn_type == 'encoder':
                st.write(get_all_attention_maps(attn_type, layers, heads, row_tokens, row_tokens, max_sentence_len, model))
            elif attn_type == 'decoder':
                st.write(get_all_attention_maps(attn_type, layers, heads, col_tokens, col_tokens, max_sentence_len, model))
            elif attn_type == 'encoder-decoder':
                st.write(get_all_attention_maps(attn_type, layers, heads, row_tokens, col_tokens, max_sentence_len, model))
    else:
        st.write('Enter a sentence to visualize the attention of the model')
    
    # add a footer with the github repo link and dataset link 
    st.markdown('---')
    st.write('Made by [Pratik Dwivedi](https://github.com/Dekode1859)')
    st.write('Check out the Scratch Implementation and Visualizer Code on [GitHub](https://github.com/Dekode1859/transformer-visualizer)')
    st.write('Dataset: [Opus-books: english-Italian](https://huggingface.co/datasets/Helsinki-NLP/opus_books)')
    # st.write('This app is a Streamlit implementation of the [BERTViz](

if __name__ == '__main__':
    main()