from flask import Flask, request, render_template,send_file
# import pickle
import pandas as pd
from thyroid.utils import load_object
from thyroid.predictor import ModelResolver
# from flask_log_helper import LogHelper
import logging
model_resolver = ModelResolver(model_registry="saved_models")
# Load the trained model, target encoder and transformer
transformer = load_object(file_path=model_resolver.get_latest_transformer_path())
model = load_object(file_path=model_resolver.get_latest_model_path())
target_encoder = load_object(file_path=model_resolver.get_latest_target_encoder_path())

print(transformer.feature_names_in_)
input_feature_name = list(transformer.feature_names_in_)

app = Flask(__name__)
static_folder='/logs'
# log_helper = LogHelper()
# log_helper.init_app(app, log_filename='app.log')



# configure logging
log_filename = 'appps.log'
handler = logging.FileHandler(log_filename)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)
app.logger.info('You are here')

@app.route('/logs')
def download_logs():
    return send_file(log_filename, as_attachment=True)







# Define a route to handle incoming requests from users
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the HTML form
    app.logger.info('You are in predict page')
    input_data = {

        'on_thyroxine': request.form['on_thyroxine'],
        'query_on_thyroxine': request.form['query_on_thyroxine'],
        'on_antithyroid_medication': request.form['on_antithyroid_medication'],
        'sick': request.form['sick'],
        'pregnant': request.form['pregnant'],
        'thyroid_surgery': request.form['thyroid_surgery'],
        'I131_treatment': request.form['I131_treatment'],
        'query_hypothyroid': request.form['query_hypothyroid'],
        'query_hyperthyroid': request.form['query_hyperthyroid'],
        'lithium': request.form['lithium'],
        'goitre': request.form['goitre'],
        'tumor': request.form['tumor'],
        'hypopituitary': request.form['hypopituitary'],
        'psych': request.form['psych'],
        'sex': request.form['sex'],
        'age': float(request.form['age']),
        'TSH': float(request.form['TSH']),
        'T3': float(request.form['T3']),
        'TT4': float(request.form['TT4']),
        'T4U': float(request.form['T4U']),
        'FTI': float(request.form['FTI'])
    }



    app.logger.info('Taken inputs')
    # Transform the input data using the target encoder and transformer
    input_df = pd.DataFrame(input_data,index=[0])
    # transformed_data = target_encoder.transform(input_df)
    app.logger.info('Converted input data to DataFrame')


    transformed_data = transformer.transform(input_df)
    app.logger.info('Transformer applied')

    # Use the transformed data as input to the model and get the predicted output
    prediction = model.predict(transformed_data)
    app.logger.info('Prediction applied')
    cat_prediction = target_encoder.inverse_transform(prediction.astype(int))
    app.logger.info('Inverse Transform applied')
    # Assign the corresponding category based on the predicted output
    app.logger.info('Assigning categories')
    if cat_prediction =='A':
        category = 'hyperthyroid'


    if cat_prediction =='A':
        category = 'hyperthyroid'
    elif cat_prediction =='B':
        category = 'T3 toxic'
    elif cat_prediction =='C':
        category = 'toxic goitre'
    elif cat_prediction =='D':
        category = 'secondary toxic'
    elif cat_prediction =='E':
        category = 'hypothyroid'
    elif cat_prediction =='F':
        category = 'primary hypothyroid'
    elif cat_prediction =='G':
        category = 'compensated hypothyroid'
    elif cat_prediction =='H':
        category = 'secondary hypothyroid'
    elif cat_prediction =='I':
        category = 'increased binding protein'
    elif cat_prediction =='J':
        category = 'decreased binding protein'
    elif cat_prediction =='K':
        category = 'concurrent non-thyroidal illness'
    elif cat_prediction =='L':
        category = 'consistent with replacement therapy'
    elif cat_prediction =='M':
        category = 'underreplaced'
    elif cat_prediction =='N':
        category = 'overreplaced'
    elif cat_prediction =='O':
        category = 'antithyroid treatment through antithyroid drugs'
    elif cat_prediction =='P':
        category = 'antithyroid treatment through I131 treatment'
    elif cat_prediction =='Q':
        category = 'antithyroid treatment through surgery'
    elif cat_prediction =='R':
        category = 'discordant assay results'
    elif cat_prediction =='S':
        category = 'elevated TBG'
    elif cat_prediction =='T':
        category = 'elevated thyroid hormones'


    elif cat_prediction =='AK':
        category = 'consistent with hypothyroidism, but there is also evidence of a concurrent non-thyroidal illness'
    elif cat_prediction =='KJ':
        category = 'consistent with hypothyroidism, but there is also evidence of concurrent non-thyroidal illness and decreased binding protein levels'
    elif cat_prediction =='MK':
        category = 'consistent with a diagnosis of underreplaced thyroid hormone in the presence of concurrent non-thyroidal illness'
    elif cat_prediction =='LJ':
        category = 'consistent with a diagnosis of hypothyroidism that is being treated with replacement therapy, but there is also evidence of decreased binding protein levels'
    elif cat_prediction =='GK':
        category = 'consistent with a diagnosis of compensated hypothyroidism in the presence of concurrent non-thyroidal illness'
    elif cat_prediction =='MI':
        category = 'consistent with a diagnosis of underreplaced thyroid hormone in the presence of increased binding protein levels'
    elif cat_prediction =='FK':
        category = 'consistent with a diagnosis of primary hypothyroidism in the presence of concurrent non-thyroidal illness.'
    elif cat_prediction =='GI':
        category = 'consistent with a diagnosis of compensated hypothyroidism in the presence of increased binding protein levels'
    elif cat_prediction =='OI':
        category = ' consistent with a diagnosis of increased binding protein levels in the presence of antithyroid drug therapy.'

    else:
        category = 'Not Suffering with Thyroidal Disease'
    app.logger.info('categories assigned')
    # Render the predicted category on the HTML template
    return render_template('result.html', category=category)


@app.route('/')
def form():
    return render_template('form.html')



# logging.basicConfig(filename='app.log', level=logging.DEBUG)




if __name__ == '__main__':
    name = 'Flask app'
    app.run(host='0.0.0.0', port=8080)


