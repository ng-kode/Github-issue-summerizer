########## Model ##########

latent_dim = 300

#### Encoder ####
encoder_inputs = Input(shape=(doc_length,), name='Encoder-Input')

x = Embedding(num_encoder_tokens, latent_dim, mask_zero=False, name='Body-Word-Embedding')(encoder_inputs)
x = BatchNormalization(name='Encoder-BatchNorm-1')(x)

_, state_h = GRU(latent_dim, return_state=True, name='Encoder-Last-GRU')(x)
_, state_h_back = GRU(latent_dim, return_state=True, go_backwards=True, name='Encoder-Last-Backward-GRU')(x)

# state_h = merge([state_h, state_h_back], mode='sum')
# att_prob = Dense(latent_dim, activation='softmax', name='attention_vec')(state_h)
# state_h = merge([state_h, att_prob], output_shape=latent_dim, name='attention_mul', mode='mul')

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


########## Result ##########
"""
Issue Body:
hi, i'm trying to get the deep link working. 
i can send the activity, open the app and read dashcode and get booleanextra and all that. 
so activating the deep link works fine and for example when i call getaction, 
it returns android.intent.action.view which is correct. 
the main problem is that getdatastring and getscheme always return null. 
i'm out of test ideas. do you think its a bug? i have attached the manifest file for your reference. 
and i'm using gvrintent.getdata that always returns null. 
islaunchedfromvr and getintenthashcode are working fine. 
and this is the command line i used to test as an example: 
./adb shell am start -w -a android.intent.action.view -d shapevisual://com.shapevisual.app?wl=gfs com.shapevisual.app androidmanifest.xml.txt https://github.com/googlevr/gvr-unity-sdk/files/864522/androidmanifest.xml.txt

>>>>> Target Title <<<<<
"android - deep link - getdatastring always returns null"

>>>>> Generated Title (From epoch 0 to 10): <<<<<
0 val_loss: NotAva - "null reference not working"
1 val_loss: 2.2540 - "null pointer in"
2 val_loss: 2.2280 - "null reference to"
3 val_loss: 2.2161 - "null pointer in"
4 val_loss: 2.2135 - "android sdk"
5 val_loss: 2.2126 - "android app crashes on startup"
6 val_loss: 2.2135 - "android app crashes when using"
7 val_loss: 2.2166 - "android app crashes when trying to load a file"
8 val_loss: 2.2232 - "android app crashes when using"
9 val_loss: 2.2279 - "android sdk"
X val_loss: 2.2349 - "null pointer exception"
"""
