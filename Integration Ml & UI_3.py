
####### Importing the libraries
import pickle
import pandas as pd
import webbrowser
# !pip install dash
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

####### Declaring Global variables
project_name =  None
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
scrappedReviews = None
ideas=['CHOOSE YOUR OPTION',"I said they are somewhat small",'Nothing like the real thing with real leather.','All you can ask for. I got what i ordered Thanks!!' ,
       'quality was good','quality was bad'," ",'Great material and perfect quality!']
####### Defining My Functions
def jaya():
        pass;
        
def loading_model():
    global scrappedReviews
    scrappedReviews = pd.read_csv('C://Users//user//Downloads//balanced_reviews.csv')
 
    global pickle_model
    file = open("C://Users/user//Downloads//pickle_model (1).pkl", 'rb') 
    pickle_model = pickle.load(file)
    
    

    global vocab
    file = open("C://Users//user//Downloads//features.pkl", 'rb') 
    vocab = pickle.load(file)

def check_review(reviewText):

    #Inputted review need to be vectorised, that vectorizer is not saved yet
    #we have pickle file, having vocab of Vectorized data we did with data set.
    #load the vectorize and call transform and then pass that to model preidctor

    transformer = TfidfTransformer()
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=vocab)
    vectorised_review = transformer.fit_transform(loaded_vec.fit_transform([reviewText]))


    return pickle_model.predict(vectorised_review)

def create_app_ui():
    global project_name
    main_layout = dbc.Container(
        dbc.Jumbotron(
                 [
                    html.H1(id = 'heading', children = project_name, className = 'display-3 mb-4',style = {'font-size':'3em','font-weight':'bold','font-family':'Roboto Black Italic'}),
                    dbc.Textarea(id = 'textarea', className="mb-3 text-center", placeholder="Enter the Review",value=" Enter your Review" , style = {'height': '170px','color':'purple','background-color':'white','font-weight':'bold'}),
                    dbc.Container([
                        dcc.Dropdown(
                    id='dropdown',
                    placeholder = 'Select a Review',
                    options=[{'label': i[:100] + "...", 'value': i} for i in ideas],
                    value = ideas[0],
                    style = {'margin-bottom': '30px'}
                    
                )
                       ],
                        style = {'padding-left': '50px', 'padding-right': '50px'}
                        ),
                    dbc.Button("Submit", color="dark", className="mt-2 mb-3", id = 'button', style = {'width': '100px'}),
                    html.Div(id = 'result'),
                    html.Div(id = 'result1')
                    ],
                className = 'text-center'
                ),
        className = 'mt-4'
        )
    
    return main_layout

def browser_opening():
    webbrowser.open_new('http://127.0.0.1:8050/')
    
@app.callback(
    Output('result', 'children'),
    [
    Input('button', 'n_clicks')
    ],
    [
    State('textarea', 'value')
    ]
    )    
def update_app_ui(n_clicks, textarea):
    result_list = check_review(textarea)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("Negative", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("Positive", color="success")
    else:
        return dbc.Alert("Unknown", color="dark")

@app.callback(
    Output('result1', 'children'),
    [
    Input('button', 'n_clicks')
    ],
    [
     State('dropdown', 'value')
     ]
    )    

def update_dropdown(n_clicks, value):
    result_list = check_review(value)
    
    if (result_list[0] == 0 ):
        return dbc.Alert("Negative", color="danger")
    elif (result_list[0] == 1 ):
        return dbc.Alert("Positive", color="success")
    else:
        return dbc.Alert("Unknown", color="dark")

####### Main Function to control the Flow of your Project
def main():
    print("Start of project")
    global project_name
    global app
    jaya()
    loading_model()
    browser_opening()
    
    
    
    project_name = "Sentiment Analysis with Insights"
    print("My project name = ", project_name)
    
    app.title = project_name
    app.layout = create_app_ui()
    app.run_server()
    
    print("End of project")
    project_name = None
    
    app = None
        
####### Calling the main function 
if __name__ == '__main__':
    main()
