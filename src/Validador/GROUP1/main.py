import argparse
import numpy as np
import pandas as pd
import sys
import joblib

def predict (input):
    """
    Generates the prediction.
    In this case, it generates a random output.
    THIS SHOULD LOAD THE MODELS INSTEAD AND GENERATE THE PREDICTIONS!

    @param input: the input dataframe.
    @return a dataframe with three columns: ID, CreditoAprobado, ScoreRiesgo
    """
    
    model_regresion = joblib.load('./GROUP1/model_regression.pkl')

    output = pd.DataFrame()
    output['Id'] = input['Id']
    score = model_regresion.predict(input)
    approved = score < 50
    output['CreditoAprobado'] = approved
    output['ScoreRiesgo'] = score
    return output


if __name__ == '__main__':

    # Creates a parser to receive the input argument.
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='Path to the data file.')
    args = parser.parse_args()

    # Read the argument and load the data.
    try:
        data = pd.read_csv(args.file)
    except:
        print("Error: the input file does not have a valid format.", file=sys.stderr)
        exit(1)

    # Computes the predictions.
    # NOTE: this stage is simulated.
    output = predict(data)

    # Writes the output.
    print('Id,CreditoAprobado,ScoreRiesgo')
    for r in output.itertuples():
        print(f'{r.Id},{r.CreditoAprobado},{r.ScoreRiesgo}')
