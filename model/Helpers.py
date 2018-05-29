import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import dill as dpickle
from keras.layers import Input, merge
from keras.models import Model

def add_start_end_(string):
    return " 'start' " + string + " 'end' "

def custom_pad_sequences(raw_sequences, num_words, maxlen_title):
    """
    Input: raw_sequences, num_words, maxlen_title
    Return: target, tk_title, word2idx, idx2word
    """
    # add _start_ and _end_
    vfunc = np.vectorize(add_start_end_)
    new_sequences = vfunc(raw_sequences)
        
    # tokenize the most frequence 4500 words
    print('tokenizing...')
    tk_title = Tokenizer()
    tk_title.fit_on_texts(new_sequences)
    tk_title.num_words = num_words
    new_sequences = tk_title.texts_to_sequences(new_sequences)
    
    # pad to length maxlen_title with _start_ & _end_ idx wrapped
    print('padding...')
    target = np.zeros((len(new_sequences), maxlen_title), dtype='int')
    for i, seq in enumerate(new_sequences):
        if len(seq) < maxlen_title:
            # direct append
            for j, idx in enumerate(seq):
                target[i][j] = idx
        else:
            # chop then add back _end_ idx
            for j, idx in enumerate(seq):
                if j < maxlen_title - 1:
                    target[i][j] = idx                    
            endidx = tk_title.word_index.get("'end'")
            target[i][-1] = endidx
    
    word2idx = tk_title.word_index    
    idx2word = {v: k for k, v in word2idx.items()}
    return target, tk_title, word2idx, idx2word

def load_encoder_inputs(encoder_np_vecs):
    """
    Input: filename
    Output: encoder_input_data, max_length_input
    """
    encoder_input_data = np.load(encoder_np_vecs)
    max_length_input = encoder_input_data.shape[1]
    print('Shape of encoder input: {}'.format(encoder_input_data.shape))
    return encoder_input_data, max_length_input

def load_decoder_inputs(decoder_np_vecs):
    """
    Input: filename
    Output: decoder_input_data, decoder_target_data
    """
    
    vectorized_title = np.load(decoder_np_vecs)
    
    # For teaching forcing
    decoder_input_data = vectorized_title[:, :-1]

    # 1 Time Step Ahea From Decoder Input Data
    decoder_target_data = vectorized_title[:, 1:]

    print('Shape of decoder input: {}'.format(decoder_input_data.shape))
    print('Shape of decoder target: {}'.format(decoder_target_data.shape))
    return decoder_input_data, decoder_target_data

def load_tokenizer(fname):
    """
    Input: filename
    Output: tokenizer object
    """
    # Load files from disk
    script_dir = os.path.dirname(__file__)
    with open(os.path.join(script_dir, fname), 'rb') as f:
        tk = dpickle.load(f)

    print('Size of vocabulary for {}: {}'.format(fname, tk.num_words))
    return tk

def create_title(body_text, tk_body, tk_title, encoder_model, decoder_model, original_title_text=None):
    max_len_body = 70
    max_len_title = 12
    word2idx_title = tk_title.word_index
    idx2word_title = {v: k for k, v in word2idx_title.items()}
        
    raw_tokenized = tk_body.texts_to_sequences([body_text])
    raw_tokenized = pad_sequences(raw_tokenized, max_len_body)
    body_encoding = encoder_model.predict(raw_tokenized)

    original_body_encoding = body_encoding
    state_value = np.array(word2idx_title["'start'"]).reshape(1, 1)        
    
    decoded_sentence = []
    stop_condition = False
    while not stop_condition:
        preds, st = decoder_model.predict([state_value, body_encoding])

        pred_idx = np.argmax(preds[:, :, 2:]) + 2

        pred_word_str = idx2word_title[pred_idx]

        if pred_word_str == "'end'" or len(decoded_sentence) >= max_len_title:
            stop_condition = True
            break
        decoded_sentence.append(pred_word_str)

        # update the decoder for the next word
        body_encoding = st
        state_value = np.array(pred_idx).reshape(1, 1)
        
    print("Issue Body:\n {} \n".format(body_text))
    if original_title_text:
        print("Original Title:\n {}".format(original_title_text))
    print("\n>>>>> Generated Title (Prediction): <<<<<\n {}".format(' '.join(decoded_sentence)))
    print('\n')
    return ' '.join(decoded_sentence)

def extract_decoder_model(model):
    latent_dim = model.get_layer('Title-Word-Embedding').output_shape[-1]
    
    decoder_inputs = model.get_layer('Decoder-Input').input
    dec_emb = model.get_layer('Title-Word-Embedding')(decoder_inputs)
    dec_bn = model.get_layer('Decoder-BatchNorm-1')(dec_emb)
    
    gru_inference_state_input = Input(shape=(latent_dim,), name='hidden_state_input')
    
    gru_out, gru_state_out = model.get_layer('Decoder-GRU')([dec_bn, gru_inference_state_input]) 
#     gru_out_back, gru_state_out_back = model.get_layer('Decoder-Backward-GRU')([dec_bn, gru_inference_state_input]) 
    
#     gru_out = merge([gru_out, gru_out_back], mode='concat')
    
    dec_bn2 = model.get_layer('Decoder-BatchNorm-2')(gru_out)
    
    dense_out = model.get_layer('Final-Output-Dense')(dec_bn2)
    
    decoder_model = Model([decoder_inputs, gru_inference_state_input],
                          [dense_out, gru_state_out])
    return decoder_model