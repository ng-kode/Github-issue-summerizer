doc_length = 70
title_length = 11
num_encoder_tokens = 8000
num_decoder_tokens = 4500


latent_dim = 300

#### Encoder ####
encoder_inputs = Input(shape=(doc_length,), name='Encoder-Input')

x = Embedding(num_encoder_tokens, latent_dim, mask_zero=False, name='Body-Word-Embedding')(encoder_inputs)
x = BatchNormalization(name='Encoder-BatchNorm-1')(x)

_, state_h = GRU(latent_dim, return_state=True, name='Encoder-Last-GRU')(x)
_, state_h_back = GRU(latent_dim, return_state=True, go_backwards=True, name='Encoder-Last-Backward-GRU')(x)

state_h = merge([state_h, state_h_back], mode='sum')
att_prob = Dense(latent_dim, activation='softmax', name='attention_vec')(state_h)
state_h = merge([state_h, att_prob], output_shape=latent_dim, name='attention_mul', mode='mul')

encoder_model = Model(encoder_inputs, state_h, name='Encoder-Model')

seq2seq_encoder_out = encoder_model(encoder_inputs)

#### Decoder ####
decoder_inputs = Input(shape=(None,), name='Decoder-Input')

x = Embedding(num_decoder_tokens, latent_dim, mask_zero=False, name='Title-Word-Embedding')(decoder_inputs)
x = BatchNormalization(name='Decoder-BatchNorm-1')(x)

decoder_gru = GRU(latent_dim, return_state=True, return_sequences=True, name='Decoder-GRU')

decoder_gru_out, _ = decoder_gru(x, initial_state=seq2seq_encoder_out)

x = BatchNormalization(name='Decoder-BatchNorm-2')(decoder_gru_out)

decoder_dense_out = Dense(num_decoder_tokens, activation='softmax', name='Final-Output-Dense')(x)

#### Seq2Seq Model ####
seq2seq_Model = Model([encoder_inputs, decoder_inputs], decoder_dense_out, name='Seq2Seq-Model')

seq2seq_Model.compile(optimizer=optimizers.Nadam(lr=1e-3), loss='sparse_categorical_crossentropy')