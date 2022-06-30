import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout, LSTM, Bidirectional
predict_msg = ["Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...",
          "Ok lar... Joking wif u oni...",
          "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"]

# Defining pre-processing hyperparameters
max_len = 50 
trunc_type = "post" 
padding_type = "post" 
oov_tok = "<OOV>" 
vocab_size = 500

train_msg = ""

tokenizer = Tokenizer(num_words = vocab_size, char_level=False, oov_token = oov_tok)
tokenizer.fit_on_texts(train_msg)

model = tf.keras.models.load_model("64x3CNN.model")

def predict_spam(predict_msg):
    new_seq = tokenizer.texts_to_sequences(predict_msg)
    padded = pad_sequences(new_seq, maxlen =max_len,
                      padding = padding_type,
                      truncating="post")
    return (model.predict(padded))

def trans_number(num):
    if num[0] > 0.89:
        print(num[0] > 0.89)
        return "This is spam"
    else:
        return "This is not spam"

import flask
from flask import request
app = flask.Flask(__name__)

app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return "<h1>Machine Learning Spam Detection</h1><p>This site is a API where you input a string and it lets you know if the string is spam or not. example: /api/v1/?id=%22Even%20my%20brother%20is%20not%20like%20to%20speak%20with%20me.%20They%20treat%20me%20like%20aids%20patent.%22.</p>"

@app.route('/api/v1/', methods=['GET'])
def api_filter():
    query_parameters = request.args

    id = query_parameters.get('id')

    results = predict_spam([id])

    output = trans_number(results[0])


    return output

app.run()