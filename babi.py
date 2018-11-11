# SINGLE OUTPUT with embedding
# MERGE seq2idf input with word level sequence  
story_maxlen = 40 
vocab_size = 3499
embed_dim =100 
import tensorflow as tf
from keras.layers import merge
RECC_NE = 64
embedding_layer = Embedding(num_words,
                            embed_dim,
                            #weights=[embedding_matrix],
                            input_length=X_train.shape[1],
                            trainable=False)


tfidf = layers.Input(shape=(story_maxlen,), dtype='int32')
emb = Embedding(story_maxlen, embed_dim,input_length = x_train.shape[1])(tfidf)
emb = Bidirectional(GRU(RECC_NE,name='GR1',recurrent_dropout=0.2))(emb)

vect = layers.Input(shape=(vocab_size,), dtype='int32')
inp = Embedding(story_maxlen, embed_dim,input_length = X_train.shape[1])(vect)
inp =  Bidirectional(GRU(RECC_NE,name='GR2',recurrent_dropout=0.2))(inp)


merged = layers.multiply([emb,inp])
droped = BatchNormalization()(merged)
droped = Dropout(0.5)(droped)
ot = Dense(200, activation = 'relu')(droped)
ot = BatchNormalization()(ot)
ot = Dropout(0.5)(ot)
otpt = Dense(2, activation='softmax')(ot)
model = Model([tfidf,vect],otpt)

model.summary()
