import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input

class Predictor():
    def __init__(self, seq2seqModel, tk_body, tk_title, maxlen_body, maxlen_title):
        self.encoder_model = self.get_encoder_model(seq2seqModel)
        self.decoder_model = self.get_decoder_model(seq2seqModel)
        self.tk_body = tk_body
        self.tk_title = tk_title
        self.maxlen_body = maxlen_body
        self.maxlen_title = maxlen_title
        
    def get_encoder_model(self, model):
        return model.get_layer('Encoder-Model')
    
    def get_decoder_model(self, model):
        latent_dim = model.get_layer('Title-Word-Embedding').output_shape[-1]
    
        decoder_inputs = model.get_layer('Decoder-Input').input
        dec_emb = model.get_layer('Title-Word-Embedding')(decoder_inputs)
        dec_bn = model.get_layer('Decoder-BatchNorm-1')(dec_emb)

        gru_inference_state_input = Input(shape=(latent_dim,), name='hidden_state_input')

        gru_out, gru_state_out = model.get_layer('Decoder-GRU')([dec_bn, gru_inference_state_input]) 

        dec_bn2 = model.get_layer('Decoder-BatchNorm-2')(gru_out)

        dense_out = model.get_layer('Final-Output-Dense')(dec_bn2)

        decoder_model = Model([decoder_inputs, gru_inference_state_input],
                              [dense_out, gru_state_out])
        return decoder_model
    
    def create_title(self, body_text, original_title_text=None):
        word2idx_title = self.tk_title.word_index
        idx2word_title = {v: k for k, v in word2idx_title.items()}

        raw_tokenized = self.tk_body.texts_to_sequences([body_text])
        raw_tokenized = pad_sequences(raw_tokenized, self.maxlen_body)
        body_encoding = self.encoder_model.predict(raw_tokenized)

        original_body_encoding = body_encoding
        state_value = np.array(word2idx_title["'start'"]).reshape(1, 1)        

        decoded_sentence = []
        stop_condition = False
        while not stop_condition:
            preds, st = self.decoder_model.predict([state_value, body_encoding])

            pred_idx = np.argmax(preds[:, :, 2:]) + 2

            pred_word_str = idx2word_title[pred_idx]

            if pred_word_str == "'end'" or len(decoded_sentence) >= self.maxlen_title:
                stop_condition = True
                break
            decoded_sentence.append(pred_word_str)

            # update the decoder for the next word
            body_encoding = st
            state_value = np.array(pred_idx).reshape(1, 1)

        return ' '.join(decoded_sentence)