import os
from model.Helpers import load_tokenizer, extract_decoder_model, create_title
from keras.models import load_model

class Predictor(object):
    def __init__(self):
        self.tk_body = load_tokenizer('tk_body.dpkl')
        self.tk_title = load_tokenizer('tk_title.dpkl')

        script_dir = os.path.dirname(__file__)
        self.seq2seq_Model = load_model(os.path.join(script_dir, './seq2seq_model_bi1_atten.h5'))

        self.encoder_model = self.seq2seq_Model.get_layer('Encoder-Model')
        self.decoder_model = extract_decoder_model(self.seq2seq_Model)

    def generate_title(self, body_text):
        title = create_title(body_text, self.tk_body, self.tk_title, self.encoder_model, self.decoder_model)
        return title